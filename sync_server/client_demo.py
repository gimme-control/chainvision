# client_demo.py
from client import submit, fetch, clear

BASE = "http://127.0.0.1:8000"

# 1) Append: keys can be numbers or strings; values can be a single "vector"
#    (list[float]) or a list of items (list[list[...]])
submit(BASE, {"9": [0.1, 0.2, 0.3]})      # append one vector to key "9"
submit(BASE, {"16": [1, 2, 3, 4]})        # create key "16" and append one item

# Clear the store (no auth; dangerous). If base_url is omitted, defaults to http://127.0.0.1:8000
clear()          # or clear(BASE)

# 2) Fetch the full snapshot
full = fetch(BASE)
print(full)