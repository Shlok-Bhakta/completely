import difflib

a = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
"""

b = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
        System.out.println("Hello World");
    }
}
"""

diff = difflib.unified_diff(a.splitlines(), b.splitlines())
print("\n".join(diff))