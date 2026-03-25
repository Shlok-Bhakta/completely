# custom diffing library where we can customize all parts of the diffing process


class Diff:
    @staticmethod
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
        a_lines = a.split("\n") if a else []
        b_lines = b.split("\n") if b else []

        prefix = 0
        while (
            prefix < len(a_lines)
            and prefix < len(b_lines)
            and a_lines[prefix] == b_lines[prefix]
        ):
            prefix += 1

        suffix = 0
        while (
            suffix < len(a_lines) - prefix
            and suffix < len(b_lines) - prefix
            and a_lines[len(a_lines) - suffix - 1] == b_lines[len(b_lines) - suffix - 1]
        ):
            suffix += 1

        a_end = len(a_lines) - suffix if suffix else len(a_lines)
        b_end = len(b_lines) - suffix if suffix else len(b_lines)

        removed = [f"- {line}" for line in a_lines[prefix:a_end]]
        added = [f"+ {line}" for line in b_lines[prefix:b_end]]

        return "\n".join(removed + added)
