# custom diffing library where we can customize all parts of the diffing process

class Diff:
    def diff_mini(a: str, b: str):
        """Will return a tiny diff of the first difference spotted. so only what changed for example

        hello world
        and 
        hellO world
        turns into
        ```diff
        - hello world
        + hellO world
        ```

        hello world
        and 
        hello world
        HI!
        turns into
        ```diff
        + HI!
        ```

        hello world
        HI!
        and 
        hello world
        turns into
        ```diff
        - HI!
        ```

        This will be used as the basis for the diff generation.
        There are a lot of weird edge cases to consider as well like lines shifting and such. 

        Parameters:
            a (str): initial string
            b (str): updated string

        Returns:
            None
        """
        for i in zip(a.split("\n"), b.split("\n")):
            print(i)
            if i[0] != i[1]:
                return f"- {i[0]}\n+ {i[1]}"
    
