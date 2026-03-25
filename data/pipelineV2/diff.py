# custom diffing library where we can customize all parts of the diffing process

class Diff:
    def diff_mini(a: str, b: str):
        """Will return a tiny diff. so only what changed for example

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
        

        Parameters:
            a (str): initial string
            b (str): updated string

        Returns:
            None
        """