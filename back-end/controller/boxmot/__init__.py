# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.46'

from controller.boxmot.postprocessing.gsi import gsi
from controller.boxmot.tracker_zoo import create_tracker, get_tracker_config
from controller.boxmot.trackers.botsort.bot_sort import BoTSORT
from controller.boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from controller.boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from controller.boxmot.trackers.hybridsort.hybridsort import HybridSORT
from controller.boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from controller.boxmot.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT",
           "create_tracker", "get_tracker_config", "gsi")
