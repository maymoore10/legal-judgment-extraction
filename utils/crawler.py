import re
import json
from datetime import time

import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3 import Retry

url = "https://supremedecisions.court.gov.il/Home/GetHtmlPage"
destPath = "resources/mixed/family/"
cases_file = open("../resources/family cases.json", encoding="utf8")
cases = json.load(cases_file)
df = []
print(len(cases["data"]))

for idx in range(0, len(cases["data"]) - 1):
    cs = cases["data"][idx]
    response = requests.post(url, {
        "path": cs["Path"],
        "fileName": cs["FileName"]
    })
    soup = BeautifulSoup(response.text)
    text = soup.get_text().replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    # df.append({"content": text, "result": "הערעור אפוא נדחה"})
    with open(destPath + cs["FileName"] + ".txt", "w", encoding="utf-8") as f:
        f.write(text)




