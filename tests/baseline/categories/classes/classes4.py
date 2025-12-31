##port
class airplane:
    def __init__(self, wingspan, length, width, cost):
        self.wingspan = wingspan
        self.length = length
        self.width = width
        self.cost = cost



class hangar:
    def __init__(capacity):
        self.capacity = capacity
        self.planes = []
    
    def add(self, plane: airplane):
        if len(self.planes) < self.capacity:
            self.planes.append(plane)
        else:
            print("No more hangar space")


class airport:
    def __init__(self, hangars, codeIATA, name):
        self.name = name
        self.codeIATA = codeIATA
        self.hangars = hangars

    def getHangar(self, hangarID):
        return self.hangars[hangarID]

    def addPlane(self, hangarID, plane):
        self.hangars[hangarID].add(plane)
    

hangar1 = hangar(10)
hangar2 = hangar(5)
hangar3 = hangar(15)

DFW = airport([hangar1, hangar2, hangar3], "DFW", "DFW International Air<cursor>")

plane = airplane(10, 10, 10, 0)
hangar1.add(plane)

plane2 = airplane(10, 10, 10, 1000)
hangar1.add(plane2)

plane3 = airplane(10, 10, 10, 2000)
hangar2.add(plane3)

plane4 = airplane(10, 10, 10, 3000)
hangar3.add(plane4)

plane5 = airplane(10, 10, 10, 4000)
DFW.addPlane(0, plane5)


