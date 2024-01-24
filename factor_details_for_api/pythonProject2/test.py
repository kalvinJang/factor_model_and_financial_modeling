from sirius import Sirius
from config_sirius import CONFIG
import pandas as pdo

sirius = Sirius(CONFIG)
sirius.download_data()
