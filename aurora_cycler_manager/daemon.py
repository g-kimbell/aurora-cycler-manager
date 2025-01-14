"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Daemon to update database, snapshot jobs and plots graphs

Updates database regularly and snapshots all jobs then analyses and plots graphs
at specified times each day. Change the update time and snapshot times in the
main block to suit your needs.
"""
import logging
import sys
import traceback
from datetime import datetime, timedelta
from time import sleep

from aurora_cycler_manager import server_manager
from aurora_cycler_manager.analysis import analyse_all_batches, analyse_all_samples
from aurora_cycler_manager.eclab_harvester import main as harvest_eclab
from aurora_cycler_manager.neware_harvester import main as harvest_neware


def handle_exceptions(func: callable) -> None:
    """Log exceptions instead of raising."""
    try:
        func()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logging.critical("Error in %s: %s", func.__name__, e)
        logging.debug(traceback.format_exc())

def daemon_loop(
        update_time: float | None = 300,
        snapshot_times: list | None = None,
        save_log: bool = False,
    ) -> None:
    """Run main loop for updating, snapshotting and plotting.

    Args:
        update_time: Time in seconds between database updates, default 300
        snapshot_times: List of times to snapshot the database each day
            specified in 24-hour format as a string, e.g. ['00:00', '12:00']
            default ['02:00']

    """
    if snapshot_times is None:
        snapshot_times = ["02:00"]

    # Set up logging
    logging.basicConfig(
        filename="aurora-daemon.log" if save_log else None,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Add a stream handler to also log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("scp").setLevel(logging.WARNING)

    if not update_time:
        update_time = 300
        logging.warning("No update time specified, defaulting to 5 minutes")
    else:
        logging.info("Sleeping for %s seconds between database updates", update_time)
    if not snapshot_times:
        snapshot_times = ["02:00"]
        logging.warning("No snapshot times specified, defaulting to 2am")
    else:
        logging.info("Snapshotting and plotting at %s each day", snapshot_times)

    now = datetime.now()
    snapshot_datetimes = [datetime.combine(now, datetime.strptime(t, "%H:%M").time()) for t in snapshot_times]
    snapshot_datetimes = [t if t > now else t + timedelta(days=1) for t in snapshot_datetimes]
    next_run_time = min(snapshot_datetimes)  # Find the earliest next run time
    logging.info("Next snapshot at %s", next_run_time)

    sm=server_manager.ServerManager()
    sm.update_db()

    # Main loop
    while True:
        sleep(update_time)
        now = datetime.now()
        logging.info("Updating database...")

        handle_exceptions(sm.update_db)

        if now >= next_run_time:
            handle_exceptions(sm.snapshot_all)
            handle_exceptions(harvest_neware)
            handle_exceptions(harvest_eclab)
            handle_exceptions(analyse_all_samples)
            handle_exceptions(analyse_all_batches)

            # Calculate the next run time for the snapshotting and analysing
            now = datetime.now()
            snapshot_datetimes = [datetime.combine(now, datetime.strptime(t, "%H:%M").time()) for t in snapshot_times]
            snapshot_datetimes = [t if t > now else t + timedelta(days=1) for t in snapshot_datetimes]
            next_run_time = min(snapshot_datetimes)
            logging.info("Next snapshot at %s", next_run_time)

def main() -> None:
    """Run the daemon, stop with KeyboardInterrupt."""
    daemon_loop()

if __name__ == "__main__":
    main()
