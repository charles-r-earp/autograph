use crate::result::Result;
use crossbeam_channel::{unbounded as unbounded_channel, Receiver, Sender};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    env,
    fs::File,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

static PROFILER: Mutex<Option<Weak<Profiler>>> = parking_lot::const_mutex(None);

#[derive(Debug)]
pub(super) struct ComputePassMetrics {
    pub(super) module_id: u32,
    pub(super) module_name: String,
    pub(super) entry_name: String,
    pub(super) invocations: usize,
    pub(super) start: Duration,
    pub(super) end: Duration,
}

enum Msg {
    #[allow(unused)]
    ComputePass(ComputePassMetrics),
}

pub(super) struct Profiler {
    #[allow(unused)]
    sender: Sender<Msg>,
}

impl Profiler {
    fn new() -> Result<Self> {
        let summary_file = File::create("autograph_profile_summary.txt")?;
        let (sender, receiver) = unbounded_channel();
        std::thread::spawn(move || run(receiver, summary_file));
        Ok(Self { sender })
    }
    fn create() -> Option<Result<Self>> {
        if let Ok(var) = env::var("AUTOGRAPH_PROFILE") {
            if var == "True" || var == "1" {
                return Some(Self::new());
            }
        }
        None
    }
    pub(super) fn get() -> Option<Result<Arc<Profiler>>> {
        let mut guard = PROFILER.lock();
        if let Some(profiler) = guard.as_ref().map(Weak::upgrade).flatten() {
            Some(Ok(profiler))
        } else {
            match Self::create() {
                Some(Ok(profiler)) => {
                    let profiler = Arc::new(profiler);
                    guard.replace(Arc::downgrade(&profiler));
                    Some(Ok(profiler))
                }
                Some(Err(err)) => Some(Err(err)),
                None => None,
            }
        }
    }
    pub(super) fn compute_pass(&self, compute_pass: ComputePassMetrics) {
        let _ = self.sender.send(Msg::ComputePass(compute_pass));
    }
}

fn run(receiver: Receiver<Msg>, mut file: File) {
    let mut summary = Summary::default();
    let mut previous_update = Instant::now();
    let mut updated = false;
    let mut disconnected = false;
    while !disconnected {
        match receiver.try_recv() {
            Ok(msg) => match msg {
                Msg::ComputePass(metrics) => {
                    let mut entry = summary
                        .compute_passes
                        .entry(ComputePassKey {
                            module_id: metrics.module_id,
                            module_name: metrics.module_name,
                            entry_name: metrics.entry_name,
                        })
                        .or_insert_with(ComputePassEntry::default);
                    entry.invocations += metrics.invocations;
                    if metrics.end > metrics.start {
                        entry.total_time += metrics.end - metrics.start;
                    }
                    updated = false;
                }
            },
            Err(err) => {
                disconnected = err.is_disconnected();
            }
        }
        if !updated && (previous_update.elapsed().as_secs() > 0 || disconnected) {
            if let Err(err) = summary.write_to_file(&mut file) {
                if cfg!(debug_assertions) {
                    panic!("{:?}", err);
                }
            } else {
                updated = true;
                previous_update = Instant::now();
            }
        }
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
struct ComputePassKey {
    module_id: u32,
    module_name: String,
    entry_name: String,
}

#[derive(Debug, Default)]
struct ComputePassEntry {
    invocations: usize,
    total_time: Duration,
}

#[derive(Default, Debug)]
struct Summary {
    compute_passes: HashMap<ComputePassKey, ComputePassEntry>,
}

impl Summary {
    fn write_to_file(&self, file: &mut File) -> std::io::Result<()> {
        use prettytable::{cell, row, Table};
        file.set_len(0)?;
        let mut table = Table::new();
        table.set_titles(row![
            "Module",
            "Entry",
            "Time %",
            "Invocations",
            "Mean Time",
            "Total Time"
        ]);
        let mut compute_passes = self.compute_passes.iter().collect::<Vec<_>>();
        compute_passes.sort_by_key(|(_, entry)| entry.total_time);
        let total = compute_passes
            .iter()
            .map(|(_, entry)| entry.total_time)
            .sum::<Duration>()
            .as_secs_f32();
        table.extend(compute_passes.iter().rev().map(|(key, entry)| {
            let percent = format!("{:.2?} %", (100. * entry.total_time.as_secs_f32()) / total);
            let mean_time = format!("{:.2?}", entry.total_time / entry.invocations as u32);
            let total_time = format!("{:.2?}", entry.total_time);
            row![
                key.module_name,
                key.entry_name,
                percent,
                entry.invocations.to_string(),
                mean_time,
                total_time
            ]
        }));
        table.print(file)?;
        Ok(())
    }
}
