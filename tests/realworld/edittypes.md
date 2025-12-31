# Different types of edits we need to track

## Basic Scenarios

```py
codehere
def foo<cursor>
codehere
```
In this scenario we only output `():` to close off the function

```py
codehere
def foo(<cursor>):
codehere
```
This one is a lil more tricky cuz now we need to think about what the args of foo are so I am thinking we output `bar, baz` and then have the editor plugin infer the `):` still being part of the func and not changing

```py
codehere
def foo(bar, baz):<cursor>
def foo(bar, bazz):
codehere
```
[NOT DOING THIS FOR NOW] In this scenario there is a syntax error because there are 2 functions and no inner content so we need to output `\ndef foo(bar, bazz):` and then have the pligin realize that the content already exists there so we need to suggest deleting it. model will need to learn this behavior also with the incomplete code modifier as well

```py
codehere
l = []
l.append(0)
l.append(1)
l.append(2)<cursor>
codehere
```
obviously there is a pattern here so the model should know to output the next `\nl.append(3)` the \n is to go down then add the `l.append(3)` same should happen with l.append(5/10/25), l.append(5/11/25), l.append(5/12/25), etc. it should be able to sense patterns and infer them

## Modifiers
Next is incomplete code
```py
codehere
def foo(bar, baz):<cursor>
```
Notice how there is no codehere after the function definition meaning this code has not been written! so model needs to have training data for this scenario. This needs to be layered on top of the above scenarios so it can complete inside the line and when there is no more context