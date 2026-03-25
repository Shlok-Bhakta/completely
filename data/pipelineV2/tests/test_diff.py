from diff import Diff
def test_diff_mini_basic_update_1():
    a = \
"""
int hi = 0;
int hi++;
cout << hi;
"""[1:-1]
    b = \
"""
int hi = 1;
int hi++;
cout << hi;
"""[1:-1]
    expected = \
"""
- int hi = 0;
+ int hi = 1;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected

def test_diff_mini_basic_remove_1():
    a = \
"""
int hi = 0;
int hi++;
cout << hi;
"""[1:-1]
    b = \
"""
int hi = 0;
cout << hi;
"""[1:-1]
    expected = \
"""
- int hi++;
"""[1:-1]
    assert Diff.diff_mini(a, b) == expected
