// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Examine overlay (Phase 3 Wave 2).
//!
//! Runs `rocm examine` — a read-only environment check — through the job-bridge
//! and shows its streamed output in the shared job console. Read-only, so it
//! needs no approval gate (the gate is for mutating actions only). This is the
//! read-only-report archetype every diagnostic screen reuses.
//!
//! The overlay also offers a Doctor report. `p` runs `rocm diagnose --report
//! --json`, which shows exactly what a report would carry and sends nothing, so
//! it needs no approval either. Only a report that parsed and was not refused
//! can be sent. `s` then stages an approval whose body is that report, and only
//! an approve runs `rocm diagnose --report --send`, which opens a prefilled mail
//! or prints the address. The dashboard builds none of this itself: it renders
//! what the binary prints, so the CLI and the dashboard cannot disagree.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use rocm_dash_core::state::{JobState, JobStatus, SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::resolve_exe;
use crate::ui::job_console::{ConsoleOutcome, on_console_key};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// The `rocm examine` job.
const EXAMINE_JOB: &str = "examine";
/// The report preview job: `rocm diagnose --report --json`. Read-only.
const REPORT_JOB: &str = "diagnose-report";
/// The send job: `rocm diagnose --report --send`. Runs only after an approve.
const SEND_JOB: &str = "diagnose-send";

/// Said when the preview output cannot be trusted. Default-deny: nothing that
/// could not be read is offered for sending.
const REPORT_UNREADABLE: &str = "The report could not be read, so nothing can be sent.";
/// Said when a refusal carries no explanation of its own.
const REPORT_REFUSED: &str = "No report was prepared for this machine.";

/// What the overlay knows about the last report preview.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReportStage {
    /// No preview has been taken since the overlay opened (or since the last one started).
    #[default]
    None,
    /// The report as the CLI printed it, one entry per line. The only stage that can be sent.
    Prepared(Vec<String>),
    /// No report can be sent, and why. The text is the CLI's own explanation when it gave one.
    Unavailable(String),
}

/// A send waiting on the user's decision.
#[derive(Debug, Clone)]
pub struct PendingReport {
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct ExamineManagerState {
    /// In-flight (or just-finished) job id: examine, the report preview, or the send.
    pub active_job: Option<String>,
    /// The last report preview.
    pub report: ReportStage,
    /// A send waiting for approve or deny. No send job exists while this is `Some`.
    pub approval: Option<PendingReport>,
}

/// Handle a key while the overlay is open. Mirrors the operational-screen seam.
pub fn on_key(
    examine: &mut Option<ExamineManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(d) = examine.as_mut() else {
        return Vec::new();
    };

    // 1) The send approval has focus. Nothing has run: approve starts the send,
    //    deny and cancel start nothing.
    if let Some(pending) = d.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                d.approval = None;
                return run_send(d, jobs);
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => d.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 2) A job is showing in the console.
    if let Some(job_id) = d.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *examine = None,
            ConsoleOutcome::Dismissed => {
                // Leaving the preview is where the output is judged, so a
                // report is classified once it is complete.
                if job_id == REPORT_JOB {
                    d.report = classify_report(jobs.job(REPORT_JOB));
                }
                d.active_job = None;
            }
            ConsoleOutcome::Unhandled => {
                let terminal = jobs.job(&job_id).is_none_or(JobState::is_terminal);
                match key.code {
                    // `r` re-runs the examine after its own terminal result.
                    KeyCode::Char('r') if terminal && job_id == EXAMINE_JOB => {
                        d.active_job = None;
                        return run_examine(d, jobs);
                    }
                    // `p` (re)prepares a report after any terminal result.
                    KeyCode::Char('p') if terminal => {
                        d.active_job = None;
                        return run_report_preview(d, jobs);
                    }
                    _ => {}
                }
            }
        }
        return Vec::new();
    }

    // 3) The intro card.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *examine = None,
        KeyCode::Enter | KeyCode::Char('r') => return run_examine(d, jobs),
        KeyCode::Char('p') => return run_report_preview(d, jobs),
        KeyCode::Char('s') => stage_send(d),
        _ => {}
    }
    Vec::new()
}

/// Judge the finished preview. Only a job that exited 0 and printed a JSON
/// object with a `schema` and no `refused` key is a report that can be sent.
/// Anything else (a failed job, text that is not JSON, an object with neither
/// key) is unreadable, which is reported as such and never offered for sending.
fn classify_report(job: Option<&JobState>) -> ReportStage {
    let unreadable = || ReportStage::Unavailable(REPORT_UNREADABLE.to_owned());
    let Some(job) = job else {
        return unreadable();
    };
    if !matches!(job.status, JobStatus::Done { code: 0 }) {
        return unreadable();
    }
    // The CLI prints the JSON object alone. Skipping anything before its first
    // line keeps a stray warning line from turning a good report unreadable.
    let lines: Vec<&str> = job
        .output
        .iter()
        .map(String::as_str)
        .skip_while(|line| !line.trim_start().starts_with('{'))
        .collect();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&lines.join("\n")) else {
        return unreadable();
    };
    let Some(object) = value.as_object() else {
        return unreadable();
    };
    if object.contains_key("refused") {
        let why = object
            .get("explanation")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .unwrap_or(REPORT_REFUSED);
        return ReportStage::Unavailable(why.to_owned());
    }
    if !object.contains_key("schema") {
        return unreadable();
    }
    ReportStage::Prepared(lines.iter().map(|line| (*line).to_owned()).collect())
}

/// Stage the send for approval. Does nothing unless a report was prepared, so
/// an unavailable or unread report cannot reach an approval. No job is started
/// here: the send exists only after an approve.
fn stage_send(d: &mut ExamineManagerState) {
    let ReportStage::Prepared(lines) = &d.report else {
        return;
    };
    let mut body = vec![
        "This is the whole of what a report would carry:".to_owned(),
        String::new(),
    ];
    body.extend(lines.iter().cloned());
    body.push(String::new());
    body.push(
        "Approving opens a prefilled mail, or shows the address and link on a machine \
         with no mail client. Nothing is sent until the mail is sent from the mail client."
            .to_owned(),
    );
    d.approval = Some(PendingReport {
        request: ApprovalRequest::new("Report this diagnosis", body),
        // The action moves machine data off this screen, so the safe button is the default.
        choice: ApprovalChoice::Deny,
    });
}

/// Run `rocm diagnose --report --json`: show what a report would carry, send nothing.
fn run_report_preview(d: &mut ExamineManagerState, jobs: &mut State) -> Vec<SideEffect> {
    d.report = ReportStage::None;
    let id = REPORT_JOB.to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: resolve_exe(),
        args: vec![
            "diagnose".to_string(),
            "--report".to_string(),
            "--json".to_string(),
        ],
    });
    d.active_job = Some(id);
    fx
}

/// Run `rocm diagnose --report --send`. Called only from an approve.
fn run_send(d: &mut ExamineManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let id = SEND_JOB.to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: resolve_exe(),
        args: vec![
            "diagnose".to_string(),
            "--report".to_string(),
            "--send".to_string(),
        ],
    });
    d.active_job = Some(id);
    fx
}

/// Build the examine overlay AND immediately start its read-only run.
///
/// Returns the overlay state plus the initial spawn side effects. The focused
/// host (bare-`rocm` `Diagnose & fix`) uses this so examine streams on open
/// without the user pressing Enter; the normal dash still opens the idle intro
/// card via [`ExamineManagerState::default`] + Enter. Reuses [`run_examine`], so
/// the spawned `cmd` resolves to the `rocm` exe with `examine` in its args.
pub fn open_running(jobs: &mut State) -> (ExamineManagerState, Vec<SideEffect>) {
    let mut d = ExamineManagerState::default();
    let fx = run_examine(&mut d, jobs);
    (d, fx)
}

/// Spawn `rocm examine` (read-only). A stable id replaces any prior console.
fn run_examine(d: &mut ExamineManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let cmd = resolve_exe();
    let id = EXAMINE_JOB.to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args: vec!["examine".to_string()],
    });
    // Examine uses a single stable id, so a no-op (a prior run still going)
    // means re-attach to that same console — intentional, unlike the
    // distinct-id screens (serve/engine/update) where a no-op surfaces an
    // "already running" message instead. Either way `active_job` points at the
    // live job, never a stale one.
    d.active_job = Some(id);
    fx
}

/// Render the overlay (intro card, or the job console once running).
pub fn draw_examine_manager(
    f: &mut Frame,
    area: Rect,
    d: &ExamineManagerState,
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Examine — environment check"),
        BoxRole::Primary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    let mut card = vec![
        Line::from(Span::styled(
            "Checks this machine's GPU, ROCm install, engines, and folders.",
            Style::default().fg(theme.fg),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Read-only — nothing is changed.",
            Style::default().fg(theme.muted),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  [ Enter: run `rocm examine` ]  ",
            Style::default()
                .bg(theme.accent)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "p: prepare a Doctor report. It is shown first, and nothing is sent.",
            Style::default().fg(theme.muted),
        )),
    ];
    match &d.report {
        ReportStage::None => {}
        ReportStage::Prepared(_) => {
            card.push(Line::from(""));
            card.push(Line::from(Span::styled(
                "A report is ready. s: review it and open a prefilled mail.",
                Style::default().fg(theme.ok),
            )));
        }
        ReportStage::Unavailable(why) => {
            card.push(Line::from(""));
            card.push(Line::from(Span::styled(
                why.clone(),
                Style::default().fg(theme.warn),
            )));
        }
    }
    f.render_widget(Paragraph::new(card).wrap(Wrap { trim: false }), rows[0]);

    let hints = if matches!(d.report, ReportStage::Prepared(_)) {
        "Enter/r run · p report · s send · Esc close"
    } else {
        "Enter/r run · p report · Esc close"
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hints,
            Style::default().fg(theme.muted),
        ))),
        rows[1],
    );

    if let Some(pending) = &d.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn enter_runs_examine_read_only_no_approval() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "spawns one job, no approval step");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        on_key(&mut d, &mut jobs, key(KeyCode::Char('q')));
        assert!(d.is_none());
    }

    #[test]
    fn r_reruns_after_a_terminal_result() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter)); // first run
        jobs.apply(StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        // `r` on a terminal job re-runs (spawns again).
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('r')));
        assert_eq!(fx.len(), 1, "r re-runs after a terminal result");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn r_at_idle_runs_examine() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('r')));
        assert_eq!(fx.len(), 1);
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn esc_closes_overlay_while_running_dismisses_when_terminal() {
        // Running: Esc closes the whole overlay (the read-only job keeps running).
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        assert!(d.as_ref().unwrap().active_job.is_some());
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.is_none(), "Esc closes the overlay while the job runs");
        // Terminal: Esc returns to the intro card (overlay stays open).
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        jobs.apply(StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.as_ref().unwrap().active_job.is_none());
        assert!(d.is_some());
    }

    #[test]
    fn snapshot_intro_then_console() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(100, 20);
        let mut term = Terminal::new(backend).unwrap();
        let d = ExamineManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_examine_manager(f, f.area(), &d, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Examine"));
        assert!(out.contains("rocm examine"));
    }

    // ── The Doctor report: review first, send only on an approve ─────────────

    /// A report as `rocm diagnose --report --json` prints it.
    const PREPARED: &[&str] = &[
        "{",
        "  \"schema\": 1,",
        "  \"architecture\": \"gfx942\",",
        "  \"entry\": \"fix-6-path\"",
        "}",
    ];

    /// A refusal envelope as the same command prints it.
    const REFUSED: &[&str] = &[
        "{",
        "  \"schema\": 1,",
        "  \"refused\": \"unreleased-hardware\",",
        "  \"explanation\": \"This machine holds hardware that is not published, so no report was prepared.\"",
        "}",
    ];

    fn spawned_args(fx: &[SideEffect]) -> Vec<Vec<String>> {
        fx.iter()
            .filter_map(|e| match e {
                SideEffect::SpawnJob { args, .. } => Some(args.clone()),
                _ => None,
            })
            .collect()
    }

    fn is_send(args: &[String]) -> bool {
        args.iter().any(|a| a == "--send")
    }

    /// Stream `lines` into job `id` and finish it with `code`.
    fn finish(jobs: &mut State, id: &str, lines: &[&str], code: i32) {
        for line in lines {
            jobs.apply(StateEvent::JobLine {
                id: id.into(),
                line: (*line).into(),
            });
        }
        jobs.apply(StateEvent::JobDone {
            id: id.into(),
            code,
        });
    }

    /// Open the overlay, run the report preview, let it finish with `lines`,
    /// and dismiss the console so the output is judged.
    fn previewed(lines: &[&str], code: i32) -> (Option<ExamineManagerState>, State) {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Char('p')));
        finish(&mut jobs, REPORT_JOB, lines, code);
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        (d, jobs)
    }

    #[test]
    fn p_prepares_a_report_with_no_approval_and_no_send() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('p')));
        assert_eq!(
            spawned_args(&fx),
            vec![vec!["diagnose", "--report", "--json"]],
            "the preview is the read-only report command, and it carries no --send"
        );
        let d = d.unwrap();
        assert_eq!(d.active_job.as_deref(), Some(REPORT_JOB));
        assert!(d.approval.is_none());
    }

    #[test]
    fn a_report_is_judged_when_its_preview_is_dismissed() {
        let (d, _) = previewed(PREPARED, 0);
        let d = d.unwrap();
        assert!(d.active_job.is_none(), "dismissing returns to the card");
        let ReportStage::Prepared(lines) = &d.report else {
            panic!("a clean report should be prepared, got {:?}", d.report);
        };
        assert_eq!(lines.len(), PREPARED.len());
        assert_eq!(lines[1], "  \"schema\": 1,");
    }

    #[test]
    fn s_stages_an_approval_that_shows_the_report_and_starts_nothing() {
        let (mut d, mut jobs) = previewed(PREPARED, 0);
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        assert!(fx.is_empty(), "staging the approval starts no job");
        let pending = d.as_ref().unwrap().approval.as_ref().expect("approval");
        assert_eq!(pending.request.title, "Report this diagnosis");
        let body = pending.request.body.join("\n");
        assert!(body.contains("\"architecture\": \"gfx942\""), "{body}");
        assert!(body.contains("Nothing is sent until"), "{body}");
        assert_eq!(
            pending.choice,
            ApprovalChoice::Deny,
            "the safe button is the default for an action that moves machine data"
        );
    }

    #[test]
    fn approve_runs_the_send_and_only_then() {
        let (mut d, mut jobs) = previewed(PREPARED, 0);
        on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(
            spawned_args(&fx),
            vec![vec!["diagnose", "--report", "--send"]]
        );
        let d = d.unwrap();
        assert_eq!(d.active_job.as_deref(), Some(SEND_JOB));
        assert!(d.approval.is_none());
    }

    #[test]
    fn deny_and_cancel_start_nothing_and_keep_the_report() {
        for decline in [
            KeyCode::Char('n'),
            KeyCode::Esc,
            KeyCode::Char('q'),
            KeyCode::Enter,
        ] {
            let (mut d, mut jobs) = previewed(PREPARED, 0);
            on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
            let fx = on_key(&mut d, &mut jobs, key(decline));
            assert!(fx.is_empty(), "{decline:?} must not start anything");
            let d = d.unwrap();
            assert!(d.approval.is_none(), "{decline:?} closes the approval");
            assert!(d.active_job.is_none());
            assert!(
                matches!(d.report, ReportStage::Prepared(_)),
                "declining leaves the report ready to review again"
            );
        }
    }

    /// The promise is that nothing is sent without an approve. A happy-path
    /// test never exercises it, so drive every other path and look for a send.
    #[test]
    fn no_send_is_ever_started_without_an_approve() {
        let mut every_effect: Vec<Vec<String>> = Vec::new();
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        // `s` before any preview, then a full preview, then every way out.
        for k in [KeyCode::Char('s'), KeyCode::Char('p')] {
            every_effect.extend(spawned_args(&on_key(&mut d, &mut jobs, key(k))));
        }
        finish(&mut jobs, REPORT_JOB, PREPARED, 0);
        for k in [
            KeyCode::Esc,
            KeyCode::Char('s'),
            KeyCode::Tab,
            KeyCode::Tab,
            KeyCode::Esc,
            KeyCode::Char('s'),
            KeyCode::Char('n'),
            KeyCode::Char('s'),
            KeyCode::Char('q'),
        ] {
            every_effect.extend(spawned_args(&on_key(&mut d, &mut jobs, key(k))));
        }
        assert!(
            every_effect.iter().all(|args| !is_send(args)),
            "a send was started without an approve: {every_effect:?}"
        );
        // The one path that does send: stage, then approve.
        on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(spawned_args(&fx).iter().filter(|a| is_send(a)).count(), 1);
    }

    #[test]
    fn a_refused_report_explains_itself_and_offers_no_send() {
        let (mut d, mut jobs) = previewed(REFUSED, 0);
        assert_eq!(
            d.as_ref().unwrap().report,
            ReportStage::Unavailable(
                "This machine holds hardware that is not published, so no report was prepared."
                    .to_owned()
            )
        );
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        assert!(fx.is_empty());
        assert!(
            d.as_ref().unwrap().approval.is_none(),
            "a refusal must never reach an approval"
        );
    }

    #[test]
    fn a_refusal_without_an_explanation_still_says_none_was_prepared() {
        let (d, _) = previewed(&["{", "  \"schema\": 1,", "  \"refused\": \"x\"", "}"], 0);
        assert_eq!(
            d.unwrap().report,
            ReportStage::Unavailable(REPORT_REFUSED.to_owned())
        );
    }

    #[test]
    fn output_that_cannot_be_read_is_never_offered_for_sending() {
        let cases: [(&str, &[&str], i32); 5] = [
            ("not json", &["this is not a report"], 0),
            (
                "no schema",
                &["{", "  \"architecture\": \"gfx942\"", "}"],
                0,
            ),
            ("an array", &["[1, 2]"], 0),
            ("a good report that exited non-zero", PREPARED, 1),
            ("no output at all", &[], 0),
        ];
        for (name, lines, code) in cases {
            let (mut d, mut jobs) = previewed(lines, code);
            assert_eq!(
                d.as_ref().unwrap().report,
                ReportStage::Unavailable(REPORT_UNREADABLE.to_owned()),
                "{name}"
            );
            on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
            assert!(d.as_ref().unwrap().approval.is_none(), "{name}");
        }
    }

    #[test]
    fn a_job_that_failed_to_run_is_never_offered_for_sending() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Char('p')));
        jobs.apply(StateEvent::JobErr {
            id: REPORT_JOB.into(),
            message: "spawn failed".into(),
        });
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert_eq!(
            d.as_ref().unwrap().report,
            ReportStage::Unavailable(REPORT_UNREADABLE.to_owned())
        );
    }

    #[test]
    fn a_stray_line_before_the_report_does_not_make_it_unreadable() {
        let mut lines = vec!["warning: a harmless notice"];
        lines.extend_from_slice(PREPARED);
        let (d, _) = previewed(&lines, 0);
        let ReportStage::Prepared(report) = d.unwrap().report else {
            panic!("a leading warning should not hide a good report");
        };
        assert_eq!(report[0], "{", "the stored report starts at the JSON");
    }

    #[test]
    fn s_with_no_prepared_report_does_nothing() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        assert!(fx.is_empty());
        assert!(d.unwrap().approval.is_none());
    }

    #[test]
    fn a_new_preview_clears_the_previous_report() {
        let (mut d, mut jobs) = previewed(PREPARED, 0);
        on_key(&mut d, &mut jobs, key(KeyCode::Char('p')));
        assert_eq!(d.unwrap().report, ReportStage::None);
    }

    #[test]
    fn p_prepares_after_the_examine_result_and_r_stays_examine_only() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter)); // examine
        finish(&mut jobs, EXAMINE_JOB, &["ok"], 0);
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('p')));
        assert_eq!(
            spawned_args(&fx),
            vec![vec!["diagnose", "--report", "--json"]]
        );
        // On the finished preview console, `r` must not re-run the examine.
        finish(&mut jobs, REPORT_JOB, PREPARED, 0);
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('r')));
        assert!(fx.is_empty(), "r is the examine re-run, not the report's");
    }

    #[test]
    fn a_pending_approval_keeps_the_overlay_from_being_ejected_by_esc() {
        let (mut d, mut jobs) = previewed(PREPARED, 0);
        on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        let mut s = crate::app::AppState::new("test".into(), "dracula".into());
        s.examine_manager = d.take();
        assert!(
            !s.active_overlay_at_root(),
            "with an approval open the overlay is not at its root, so Esc cancels the approval first"
        );
    }

    fn render(d: &ExamineManagerState) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        let jobs = State::default();
        term.draw(|f| draw_examine_manager(f, f.area(), d, &jobs, &theme))
            .unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn the_card_always_advertises_the_report_and_offers_send_only_when_prepared() {
        let idle = render(&ExamineManagerState::default());
        assert!(idle.contains("prepare a Doctor report"), "{idle}");
        assert!(!idle.contains("s send"), "no send hint without a report");

        let (d, _) = previewed(PREPARED, 0);
        let ready = render(&d.unwrap());
        assert!(ready.contains("A report is ready"), "{ready}");
        assert!(ready.contains("s send"), "{ready}");

        let (d, _) = previewed(REFUSED, 0);
        let refused = render(&d.unwrap());
        assert!(refused.contains("no report was prepared"), "{refused}");
        assert!(!refused.contains("s send"), "no send hint after a refusal");
        assert!(!refused.contains("A report is ready"));
    }

    #[test]
    fn the_approval_shows_the_report_and_the_promise() {
        let (mut d, mut jobs) = previewed(PREPARED, 0);
        on_key(&mut d, &mut jobs, key(KeyCode::Char('s')));
        let out = render(&d.unwrap());
        assert!(out.contains("Report this diagnosis"), "{out}");
        assert!(out.contains("gfx942"), "{out}");
        assert!(out.contains("Nothing is sent until"), "{out}");
    }
}
