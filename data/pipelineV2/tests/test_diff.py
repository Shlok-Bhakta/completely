from diff import Diff


def test_diff_mini_basic_update_1():
    a = """
int hi = 0;
int hi++;
cout << hi;
"""[1:-1]
    b = """
int hi = 1;
int hi++;
cout << hi;
"""[1:-1]
    expected = """
- int hi = 0;
+ int hi = 1;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected


def test_diff_mini_basic_remove_1():
    a = """
int hi = 0;
int hi++;
cout << hi;
"""[1:-1]
    b = """
int hi = 0;
cout << hi;
"""[1:-1]
    expected = """
- int hi++;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected


def test_diff_mini_basic_insert_1():
    a = """
int hi = 0;
cout << hi;
"""[1:-1]
    b = """
int hi = 0;
int hi++;
cout << hi;
"""[1:-1]
    expected = """
+ int hi++;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected


def test_diff_mini_append_at_end():
    a = """
int hi = 0;
cout << hi;
"""[1:-1]
    b = """
int hi = 0;
cout << hi;
return 0;
"""[1:-1]
    expected = """
+ return 0;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected


def test_diff_mini_identical_returns_empty():
    a = """
int hi = 0;
cout << hi;
"""[1:-1]
    b = a
    assert Diff.diff_mini(a, b) == ""


def test_diff_mini_multiline_replacement_block():
    a = """
setup();
int hi = 0;
int hi++;
cout << hi;
done();
"""[1:-1]
    b = """
setup();
int hi = 1;
int hi += 2;
cout << hi;
done();
"""[1:-1]
    expected = """
- int hi = 0;
- int hi++;
+ int hi = 1;
+ int hi += 2;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected


def test_diff_mini_insert_in_middle_preserves_suffix():
    a = """
line 1
line 2
line 3
"""[1:-1]
    b = """
line 1
new line
line 2
line 3
"""[1:-1]
    expected = """
+ new line
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected
