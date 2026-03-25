# Different types of edits we need to track

## Format
We need something decently simple for a small model to pick up on and generate

before edit:
```python
def sum(a, b):
    x = a + b
    return x
```
after edit:
```python
def sum(a, b):
    return x
```

so after we go and make our "change" we can create a diff that looks a little bit like this:

```diff
ⵒ
- x = a + b
```
This was chosen because we want some unique token to delimit the diffs for now its `ⵒ` but realistically can be anything unique.

a full trace looks like
```diff
ⵒ
- x = a + b
ⵒ
- return x
+ return a + b
```

this way the model can use the past to predict the future


## Scenarios

### Changing from one line to another
```python
from numpy import np
```
this kinda thing is what would make diffing good cuz it would know what I am trying to do and if this was at the top of the file then it would know!
a diff sequence of this would look something like:
```diff
ⵒ
- import numpy
+ from numpy import np
```
now we need some way to generalize this out to any code any language

maybe we can do a simple typo

```python
print("Hamburger")
```
and we can typo this to 
```python
pritn("Hamburger")
```
or 
```python
printn("Hamburger")
```
This can get us some level of correction

```diff
ⵒ
- pritn("Hamburger")
+ print("Hamburger")
```

this will show the model what the diff format looks like and such also easy to code. we can save the input case and stuff for fine tuning for pretraining this is probably the play.

### Writing code 

#### At the end of a file
Maybe 20% of the time we can pick a random spot in the file and simulate writing new code.

```python
cases = int(input())
for i in range(cases):
    input()
    input()
    sg = max([int(x) for x in input().split(" ")])
    sm = max([int(x) for x in input().split(" ")])
    if(sg >= sm):
        print("Godzilla")
    elif(sm > sg):
        print("MechaGodzilla")
```

```python
cases = int(input())
for i in range(cases):
    input()
    input()
    sg = max([int(x) for x in input().split(" ")])
```

we can even go further and go mid line
```python
cases = int(input())
for i in range(cases):
    input()
    input()
    sg = max([int(x) for x in input
```

we can then create a system for reintroducing some the diff parts

```diff
ⵒ
+     im = max([int(x) for x in input().split(" ")])
ⵒ
+     if(sg >= sm):
ⵒ
+         print("Godzilla")
ⵒ
+     elif(sm > sg):
```
Showing a series of code buing built and as a bonus we can generate 2 or 3 of them during inference to rapid fire the next output if the user accepts.

and for the mid line example we can do
```diff
ⵒ
-     sg = max([int(x) for x in input
+     sg = max([int(x) for x in input().split(" ")])
ⵒ
+     im = max([int(x) for x in input().split(" ")])
ⵒ
+     if(sg >= sm):
ⵒ
+         print("Godzilla")
ⵒ
+     elif(sm > sg):
```


#### At the middle/start of a file
Maybe rest of the time we can pick a random spot in the file and carve out a spot and simulate filling it in. 

```python
cases = int(input())
for i in range(cases):
    input()
    input()
    sg = max([int(x) for x in input().split(" ")])
    sm = max([int(x) for x in input().split(" ")])
    if(sg >= sm):
        print("Godzilla")
    elif(sm > sg):
        print("MechaGodzilla")
```




