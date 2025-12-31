# This is gonna use openrouter and expose a way to try conpletions with it
from openrouter import OpenRouter

class OpenRouterAdapter:
    def __init__(self, model):
        self.model = model
        import dotenv
        import os
        dotenv.load_dotenv()
        self.openrouter = OpenRouter(os.getenv("OPENROUTER_API_KEY"))
    
    def request_completion(self, fim, prompt="""<instructions>
You are a code infilling model.
Given PREFIX and SUFFIX, produce ONLY the code that should be inserted between them.
Do not repeat any PREFIX or SUFFIX code. Do not add comments or prose.

<PRE>
import sqlite3

conn = sqlite3.connect(':memory:')
c = conn.cursor()

c.execute('''CREATE TABLE users
             (id INTEGER PRIMARY KEY, name TEXT, age  
<SUF>
>)''')

c.execute("INSERT INTO users VALUES (?, ?, ?)",
          (1, "John", 30)) 
<MID>
</instructions>"""):
        completion = self.openrouter.chat.send(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": fim},
            ]
        )
        return completion.choices[0].message.content
