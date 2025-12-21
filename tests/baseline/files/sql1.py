##INTEGER
import sqlite3

conn = sqlite3.connect(':memory:')
c = conn.cursor()

c.execute('''CREATE TABLE users
             (id INTEGER PRIMARY KEY, name TEXT, age <cursor>)''')

c.execute("INSERT INTO users VALUES (?, ?, ?)",
          (1, "John", 30))