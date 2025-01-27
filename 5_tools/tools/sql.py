import sqlite3
import os
# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dbpath = os.path.join(current_dir, "db.sqlite")
print(dbpath)

conn = sqlite3.connect(dbpath)

# def run_sqlite_query(query):
#     cursor = conn.cursor()
#     cursor.execute(query)
#     return cursor.fetchall()

# res = run_sqlite_query("SELECT * FROM users")
# print(res)


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

print(list_tables())