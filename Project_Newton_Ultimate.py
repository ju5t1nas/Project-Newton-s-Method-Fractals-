from scipy import*
from scipy.linalg import *
from matplotlib.pyplot import *
import time

#Project Fractals a.k.a. Pretty Pictures
#by Justinas Smertinas, Aiste Remeikaite, Erik Berggren, Fredrik Hansson Heidi Mach, Hogo Lövden.

#Here we define a class Fractal2D.
#Fractal2D takes in the two functions and their partial derivatives as inputs. Partial derivatives are optional.
class Fractal2D:
    #We initalise Fractal 2D.
    #You will notice that gder2 is predefined to be equal to true.
    #We will later explain the reason for this is to make sure no input was missed. Otherwise approximation of all derivatives will be forced.
    def __init__(self, Ffunc, Gfunc, fder1, fder2, gder1, gder2 = True):

            # F function
        self.Ffunc = Ffunc
            # F function
        self.Gfunc = Gfunc
            # F partial derivative with respect to the first variable
        self.fder1 = fder1
            # F partial derivative with respect to the second variable
        self.fder2 = fder2
            # G partial derivative with respect to the first variable
        self.gder1 = gder1
            # F partial derivative with respect to the second variable
        self.gder2 = gder2
            #This is a list of roots that values may converge to. Used in Task 3 to store values.
            #At the start the only item in the list is None, which we use to represent non-convergence.
        self.zeroes = [None]
            #This is a list of how many itterations it took for a value to converge to.
            #It is used in Task 7 in a similar way as list of roots above.
        self.IterNum = [0,]

            #Here is where gder2 = True comes in.
            #If gder2 == True, this means that user has missed one of partial derivatives.
            #This means that we must approximate all the derivatives. Otherwise errors occur.
        if self.gder2 == True:
            #We set a parameter Approx, which will be checked be later methods.
            #If self.Approx == True, then we force approximation.
            self.Approx = True
        else:
            self.Approx = False

    #This is one of the Newton's Methods we have defined.
    #It does not use inverses or any other scipy commands.
    #It works only for two variables, but is Super Fast.
    #More information about this Method can be found in the book
    #"Calculus a Complete Course 8th Edition" by Robert.A.Adams Chapter
    def NewtonBad(self, guess, Approx = True):

        #We create to "holder" variables x and y for finding approximating the derivative.
        #and seeing whether the values are converging.
        x = guess[0]
        y = guess[1]

        #We check whether User wants to approximate.
        if Approx == True:

            #We create a for loop to loop throught the method.
            for i in range(1000):
                #We calculate the a values of partial derivatives
                #using the definition of the derivative for small values of h.
                fder1 = (self.Ffunc([x + 1.e-10, y]) - self.Ffunc([x, y])) / (1.e-10)
                fder2 = (self.Ffunc([x, y + 1.e-10]) - self.Ffunc([x, y])) / (1.e-10)
                gder1 = (self.Gfunc([x + 1.e-10, y]) - self.Gfunc([x, y])) / (1.e-10)
                gder2 = (self.Gfunc([x, y + 1.e-10]) - self.Gfunc([x, y])) / (1.e-10)

                #We create a point XY so it is easie to plug in [x, y].
                #For lazyness sake.
                XY = [x, y]

                #We compute the values of numerators and denominators of values required to change X and Y
                #We essentially manually compute the determinants
                num1 = (self.Ffunc(XY) * gder2 - fder2 * self.Gfunc(XY))
                num2 = (fder1 * self.Gfunc(XY) - self.Ffunc(XY) * gder1)
                dem = (fder1 * gder2 - fder2 * gder1)

                #Here we need to be carefull that Denominator is not 0 as it means divergence.
                if dem != 0:
                    #Here we create xNew and yNew to be later compared to x and y
                    xNew = x - num1/dem
                    yNew = y - num2/dem
                    #If abs of the differnece between (xNew and x) and (yNew and y) are both below desired threshold we say they converge.
                    if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                        x = xNew
                        y = yNew
                        #Then we return number to which they converged.
                        return XY
                    #If max number of iterations is reached we say that the point does not converge.
                    elif i == 999:
                        #We thus return None which we use to symbolise non-convergence.
                        return None
                    #Else set new x and y as old and repeat the process
                    else:
                        x = xNew
                        y = yNew
                #If denominator = 0, we see divergence.
                else:
                    return None


        #We make sure we can do non-approximation
        elif Approx == False and self.Approx == False:
            #Basically the same as above, but uses predefined partial derivatives to compute their values.
            for i in range(1000):
                XY = [x, y]
                num1 = (self.Ffunc(XY)*self.gder2(XY) - self.fder2(XY)*self.Gfunc(XY))
                num2 = (self.fder1(XY)*self.Gfunc(XY) - self.Ffunc(XY)*self.gder1(XY))
                dem = (self.fder1(XY)*self.gder2(XY) - self.fder2(XY)*self.gder1(XY))
                if dem != 0:
                    xNew = x - num1/dem
                    yNew = y - num2/dem
                    if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                        x = xNew
                        y = yNew
                        return XY
                    elif i == 999:
                        return None
                    else:
                        x = xNew
                        y = yNew
                else:
                    return None

        #Else we approximate.
        #Same as if.
        #Skip to the next method.
        else:

            for i in range(1000):
                fder1 = (self.Ffunc([x + 1.e-10, y]) - self.Ffunc([x, y])) / (1.e-10)
                fder2 = (self.Ffunc([x, y + 1.e-10]) - self.Ffunc([x, y])) / (1.e-10)
                gder1 = (self.Gfunc([x + 1.e-10, y]) - self.Gfunc([x, y])) / (1.e-10)
                gder2 = (self.Gfunc([x, y + 1.e-10]) - self.Gfunc([x, y])) / (1.e-10)
                XY = [x, y]
                num1 = (self.Ffunc(XY) * gder2 - fder2 * self.Gfunc(XY))
                num2 = (fder1 * self.Gfunc(XY) - self.Ffunc(XY) * gder1)
                dem = (fder1 * gder2 - fder2 * gder1)
                if dem != 0:
                    xNew = x - num1/dem
                    yNew = y - num2/dem
                    if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                        x = xNew
                        y = yNew
                        return XY
                    elif i == 999:
                        return None
                    else:
                        x = xNew
                        y = yNew
                else:
                    return None

    #This is the proper Newton's method which is not as fast but could scales for more variables.
    def NewtonGood(self, guess, Approx = True):

        #We create x and y variables as in previous method
        x=guess[0]
        y=guess[1]

        #We update guest with new values of x and y.
        guess = [x,y]

        #If user wishes to approximate, we approximate.
        if Approx == True:

            #We define an approximation for the first partial derivative.
            def xdiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x + h, y]) - fun([x, y])) / h

            #We define an approximation for the second partial derivative.
            def ydiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x, y + h]) - fun([x, y])) / h

            #We create a loop to loop trought the process.
            for i in range(100):
                #We create a Jacobian matrix.
                Jacob = []
                Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                Jacob = array(Jacob)

                #We impose some condition on the determinants on the Jacobian matrix.
                if det(Jacob) == 0 and abs(self.Ffunc(guess)) < 1.e-6 and abs(self.Gfunc(guess)) < 1.e-9:
                    return guess
                elif det(Jacob) == 0:
                    return None
                #If the loop is exhausted, we return that there was no convergence
                elif i == 99:
                    return None
                Jacob = array(Jacob)
                guess = array(guess)
                guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                xNew = guess[0]
                yNew = guess[1]
                #We check whether the values of x and y converge
                if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                    return [xNew, yNew]

                #We update the values of x and y to be the newly computed x and y
                x = xNew
                y = yNew

        #We check whether the user wants to approximate and whether we can allow to approximate.
        elif Approx == False and self.Approx == False:
            #We create another for loop the same as in the approximation version above.
            for i in range(100):
                Jacob = []
                #Here we use predefined partial derivatives to construct the Jacobian martix.
                #The rest of is the same as the version above.
                Jacob.append(array([self.fder1(guess), self.fder2(guess)]))
                Jacob.append(array([self.gder1(guess), self.gder2(guess)]))
                Jacob = array(Jacob)
                if det(Jacob) == 0 and abs(self.Ffunc(guess)) < 1.e-6 and abs(self.Gfunc(guess)) < 1.e-9:
                    return guess
                elif det(Jacob) == 0:
                    return None
                elif i == 99:
                    return None
                Jacob=array(Jacob)
                guess=array(guess)
                guess=guess-dot(inv(Jacob), array([self.Ffunc([x,y]),self.Gfunc([x,y])]))
                xNew=guess[0]
                yNew=guess[1]
                if abs(x-xNew) <= 1.e-6 and abs(y-yNew) <= 1.e-6:
                    return [xNew, yNew]

                x = xNew
                y = yNew
        #Else we approximate
        else:
            #Here we do the same as in the when we approximated at first.
            def xdiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x + h, y]) - fun([x, y])) / h

            def ydiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x, y + h]) - fun([x, y])) / h

            for i in range(100):
                Jacob = []
                Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                Jacob = array(Jacob)
                if det(Jacob) == 0 and abs(self.Ffunc(guess)) < 1.e-6 and abs(self.Gfunc(guess)) < 1.e-9:
                    return guess
                elif det(Jacob) == 0:
                    return None
                elif i == 99:
                    return None
                Jacob = array(Jacob)
                guess = array(guess)
                guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                xNew = guess[0]
                yNew = guess[1]
                if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                    return [xNew, yNew]

                x = xNew
                y = yNew

    #This is the Simplified version of the Newton's method. The jacobian is only computed once.
    #Even though the Jacobian is computed only once, and thus less computations are done,
    #we have found that this drastically increases the amount of iterations needed to converge. We are talking north of tenfold.
    def NewtonSimplified(self, guess, Approx = True):

        #We again createa x and y separate variables
        x=guess[0]
        y=guess[1]

        #If user wishes to approximate, we approximate.
        if Approx == True:

            #As previously we define derivative approximation.
            def xdiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x + h, y]) - fun([x, y])) / h

            def ydiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x, y + h]) - fun([x, y])) / h

            #We construct the Jacobian matrix, but this time outside the for loop
            Jacob = []
            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
            if det(Jacob) == 0 and abs(self.Ffunc(guess)) < 1.e-6 and abs(self.Gfunc(guess)) < 1.e-9:
                return guess
            elif det(Jacob) == 0:
                return None

            #We create a loop with max range of 9999. This high number is required as the values converge really slowly.
            for i in range(10000):
                if i == 9999:
                    return None
                Jacob = array(Jacob)
                guess = array(guess)
                guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                xNew = guess[0]
                yNew = guess[1]
                #Again, we set a tolerance
                if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                    return [xNew, yNew]

                #And update x and y
                x = xNew
                y = yNew

        #We again check whether the user wants to approximate and whether we allow to approximate.
        elif Approx == False and self.Approx == False:

            #Things here are basically the same as previously except we do not
            Jacob = []
            Jacob.append(array([self.fder1(guess), self.fder2(guess)]))
            Jacob.append(array([self.gder1(guess), self.gder2(guess)]))
            if det(Jacob) == 0 and abs(self.Ffunc(guess)) < 1.e-6 and abs(self.Gfunc(guess)) < 1.e-9:
                return guess
            elif det(Jacob) == 0:
                return None

            for i in range(10000):

                Jacob=array(Jacob)
                guess=array(guess)
                guess=guess-dot(inv(Jacob), array([self.Ffunc([x,y]),self.Gfunc([x,y])]))
                xNew=guess[0]
                yNew=guess[1]
                if abs(x-xNew) <= 1.e-6 and abs(y-yNew) <= 1.e-6:
                    return [xNew, yNew]
                elif i == 9999:
                    return None
                x = xNew
                y = yNew

        #Else we approximate.
        else:

            def xdiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x + h, y]) - fun([x, y])) / h

            def ydiff(fun, guess):
                x = guess[0]
                y = guess[1]
                h = 1.e-6
                return (fun([x, y + h]) - fun([x, y])) / h

            for i in range(1000):
                Jacob = []
                Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                if det(Jacob) == 0:
                    return guess
                elif i == 999:
                    return None
                Jacob = array(Jacob)
                guess = array(guess)
                guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                xNew = guess[0]
                yNew = guess[1]
                if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                    return [xNew, yNew]

                x = xNew
                y = yNew

    #The next method does the following tasks
    # 1: Apply Newtons Method to a guess to obtain a root.
    # 2: Adds New Roots after comparing to previous Zeroes or
    # 3: Returns the number of the Root in the self.zeroes list.
    def NewZero(self, guess, NewtonType = 1, Approx = True):

        #We allow to chose the prefered Newton Method by the user.
        #We set the default as 1, because it is by far the fastest.
        #However, if user enters something except 1 or 2, the method uses the Proper Newthon's method (NewtonGood)
        if NewtonType == 1:
            #We create a varible Zero, which we are going to later compare to the list of Roots self.zeroes
            Zero = self.NewtonBad(guess, Approx)

        elif NewtonType == 2:
            #We do the same here.
            Zero = self.NewtonSimplified(guess, Approx)

        else:
            #And here
            Zero = self.NewtonGood(guess, Approx)

        #We compare our Zero to every known root in self.zeroes
        for i in self.zeroes:
            #First we check whether Zero is not None - a indicator of divergence
            if type(Zero) == type(None):
                #If it is, then we return 0, as None's index in self.zeroes is 0
                return 0
            #We check whether Zero is a list and i we picked from self.zeroes is a list.
            elif type(Zero) == list and type(i) == list:
                #We check whether Zero and i are close within a tolerance from each other.
                #If so, we treat them as the same zero.
                #We use a tolerance of 1.e-4 as smaller tolerances mess with the simplified version of Newton's Method.
                if abs(Zero[0]-i[0]) <= 1.e-4 and abs(Zero[1]-i[1]) <= 1.e-4:
                    #If we decide that Zero and i are the same, we return index of i in self.zeroes
                    return self.zeroes.index(i)
                #We check whether i is the last item in the self.zeroes list.
                elif self.zeroes.index(i)+1 == len(self.zeroes):
                    #If it is, we add Zero to self.zeroes and return it's index.
                    self.zeroes.append(Zero)
                    return self.zeroes.index(Zero)
            #We check whether self.zeroes has only one item, then we know to automatically add Zero to the list.
            elif len(self.zeroes) == 1:
                self.zeroes.append(Zero)
                return self.zeroes.index(Zero)

    #The following method generates a visualisation of which points converged to which root.
    #In other words generates the fractal image.
    def Plot(self, N, a, b, c, d, Approx = True):
        #Using a and b we create a list of N number of values between a and b for x-axis.
        xvalues = linspace(a, b, N)
        #Using a and b we create a list of N number of values between c and d for x-axis.
        yvalues = linspace(c, d, N)
        #Using meshgrid() we create two list to create a grid.
        [xx, yy] = meshgrid(xvalues, yvalues)

        #We ask user for which Newton Method Type is desired
        NewtonType = input("Choose a Newton's Method")

        #We create a matrix, in which we store indexes of to which root did the guess converge.
        #[i][j]th number in the matrix corresponds to the [i][j]th point on the grid.
        ZeroIndexMatrix = zeros((N,N))

        #Here we use who fro loops to go through all points on the grind and find to which root they converge.
        for i in range(N):
            for j in range(N):
                #We pick a point.
                Point = array([xx[i][j], yy[i][j]]).T
                #We print the Point to see at which point the method is at. Sort of like a loading bar.
                print(Point)
                #We replace the [i][j]th zero on the patrix with the Point's convergene index
                ZeroIndexMatrix[i][j] = self.NewZero(Point, NewtonType, Approx)

        #We then use pcolor to put the grid and the matrix together and produce a graph.
        pcolor(xx, yy, ZeroIndexMatrix)
        #We use show() to show the graph in a separate window for compatibility with different IDEs
        show(pcolor)

    #The following method generates a visualistation of how many iterations it took to find to where a point converges.
    #In other words, a fractal image with more different colours
    def PlotIter(self, N, a, b, c, d, Approx = True):
        #We start identically as in Plot() method
        xvalues = linspace(a, b, N)
        yvalues = linspace(c, d, N)
        [xx, yy] = meshgrid(xvalues, yvalues)

        #Again, we allow to choose which Newton's method to use.
        NewtonType = input("Choose a Newton's Method! [1, 2 or 3]")

        #We create a matrix with a similar purpose as in previous Method.
        #However this one will store the number of iterations per point
        IterIndexMatrix = zeros((N,N))

        #We define a function NewInter, which has a very basically NewZero method,
        #but return index of the number number of iterations it took for a guess to converge.
        #There are a couple of differences, however they are not importand, so feel free to skip it.
        def NewIter(guess, NewtonType, Approx=True):

            # Works
            if NewtonType == 1:
                def NewtonBad1(guess, Approx):

                    x = guess[0]
                    y = guess[1]

                    if Approx == True:


                        for i in range(1000):
                            fder1 = (self.Ffunc([x + 1.e-10, y]) - self.Ffunc([x, y])) / (1.e-10)
                            fder2 = (self.Ffunc([x, y + 1.e-10]) - self.Ffunc([x, y])) / (1.e-10)
                            gder1 = (self.Gfunc([x + 1.e-10, y]) - self.Gfunc([x, y])) / (1.e-10)
                            gder2 = (self.Gfunc([x, y + 1.e-10]) - self.Gfunc([x, y])) / (1.e-10)
                            XY = [x, y]
                            num1 = (self.Ffunc(XY) * gder2 - fder2 * self.Gfunc(XY))
                            num2 = (fder1 * self.Gfunc(XY) - self.Ffunc(XY) * gder1)
                            dem = (fder1 * gder2 - fder2 * gder1)
                            if dem != 0:
                                xNew = x - num1 / dem
                                yNew = y - num2 / dem
                                if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                                    x = xNew
                                    y = yNew
                                    return i + 2
                                elif i == 999:
                                    return 1
                                else:
                                    x = xNew
                                    y = yNew
                            else:
                                return 1

                    elif Approx == False and self.Approx == False:

                        for i in range(1000):
                            XY = [x, y]
                            num1 = (self.Ffunc(XY) * self.gder2(XY) - self.fder2(XY) * self.Gfunc(XY))
                            num2 = (self.fder1(XY) * self.Gfunc(XY) - self.Ffunc(XY) * self.gder1(XY))
                            dem = (self.fder1(XY) * self.gder2(XY) - self.fder2(XY) * self.gder1(XY))
                            if dem != 0:
                                xNew = x - num1 / dem
                                yNew = y - num2 / dem
                                if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                                    x = xNew
                                    y = yNew
                                    return i + 2
                                elif i == 999:
                                    return 1
                                else:
                                    x = xNew
                                    y = yNew
                            else:
                                return 1

                    else:

                        for i in range(1000):
                            fder1 = (self.Ffunc([x + 1.e-10, y]) - self.Ffunc([x, y])) / (1.e-10)
                            fder2 = (self.Ffunc([x, y + 1.e-10]) - self.Ffunc([x, y])) / (1.e-10)
                            gder1 = (self.Gfunc([x + 1.e-10, y]) - self.Gfunc([x, y])) / (1.e-10)
                            gder2 = (self.Gfunc([x, y + 1.e-10]) - self.Gfunc([x, y])) / (1.e-10)
                            XY = [x, y]
                            num1 = (self.Ffunc(XY) * gder2 - fder2 * self.Gfunc(XY))
                            num2 = (fder1 * self.Gfunc(XY) - self.Ffunc(XY) * gder1)
                            dem = (fder1 * gder2 - fder2 * gder1)
                            if dem != 0:
                                xNew = x - num1 / dem
                                yNew = y - num2 / dem
                                if abs(x - xNew) <= 1.e-9 and abs(y - yNew) <= 1.e-9:
                                    x = xNew
                                    y = yNew
                                    return i + 2
                                elif i == 999:
                                    return 1
                                else:
                                    x = xNew
                                    y = yNew
                            else:
                                return 1

                Iter = NewtonBad1(guess, Approx)

            elif NewtonType == 2:
                def NewtonSimplified1(guess, Approx=True):

                    x = guess[0]
                    y = guess[1]

                    if Approx == True:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        Jacob = []
                        Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                        Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                        if det(Jacob) == 0:
                            return 1

                        for i in range(10000):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 9999:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                    elif Approx == False and self.Approx == False:

                        Jacob = []
                        Jacob.append(array([self.fder1(guess), self.fder2(guess)]))
                        Jacob.append(array([self.gder1(guess), self.gder2(guess)]))
                        if det(Jacob) == 0:
                            return 1

                        for i in range(10000):

                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1
                            elif i == 9999:
                                return 0
                            x = xNew
                            y = yNew


                    else:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        for i in range(1000):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 999:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                    x = guess[0]
                    y = guess[1]

                    if Approx == True:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        Jacob = []
                        Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                        Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                        if det(Jacob) == 0:
                            return 1

                        for i in range(1000):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 999:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                    elif Approx == False and self.Approx == False:

                        Jacob = []
                        Jacob.append(array([self.fder1(guess), self.fder2(guess)]))
                        Jacob.append(array([self.gder1(guess), self.gder2(guess)]))
                        if det(Jacob) == 0:
                            return 1

                        for i in range(100):
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1
                            elif i == 99:
                                return 0
                            x = xNew
                            y = yNew


                    else:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        for i in range(1000):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 999:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                Iter = NewtonSimplified1(guess, Approx)

            else:
                def NewtonGood1(guess, Approx=True):

                    x = guess[0]
                    y = guess[1]

                    guess = [x, y]

                    if Approx == True:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        for i in range(100):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            Jacob = array(Jacob)

                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 99:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                    elif Approx == False and self.Approx == False:

                        for i in range(100):
                            Jacob = []
                            Jacob.append(array([self.fder1(guess), self.fder2(guess)]))
                            Jacob.append(array([self.gder1(guess), self.gder2(guess)]))
                            Jacob = array(Jacob)
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 99:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                    else:

                        def xdiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x + h, y]) - fun([x, y])) / h

                        def ydiff(fun, guess):
                            x = guess[0]
                            y = guess[1]
                            h = 1.e-6
                            return (fun([x, y + h]) - fun([x, y])) / h

                        for i in range(100):
                            Jacob = []
                            Jacob.append(array([xdiff(self.Ffunc, guess), ydiff(self.Ffunc, guess)]))
                            Jacob.append(array([xdiff(self.Gfunc, guess), ydiff(self.Gfunc, guess)]))
                            Jacob = array(Jacob)
                            if det(Jacob) == 0:
                                return i + 1
                            elif i == 99:
                                return 0
                            Jacob = array(Jacob)
                            guess = array(guess)
                            guess = guess - dot(inv(Jacob), array([self.Ffunc([x, y]), self.Gfunc([x, y])]))
                            xNew = guess[0]
                            yNew = guess[1]
                            if abs(x - xNew) <= 1.e-6 and abs(y - yNew) <= 1.e-6:
                                return i + 1

                            x = xNew
                            y = yNew

                Iter = NewtonGood1(guess, Approx)

            if Iter in self.IterNum:
                return self.IterNum.index(Iter)
            else:
                self.IterNum.append(Iter)

        #Just as in the Plot() method, we loop throught the grid using two for loops.
        for i in range(N):
            for j in range(N):
                Point = array([xx[i][j], yy[i][j]]).T
                #We print the point for which the operations are being run to monitor how many are left to do.
                print(NewIter(Point, NewtonType, Approx))
                IterIndexMatrix[i][j] = NewIter(Point, NewtonType, Approx)

        print(IterIndexMatrix)
        #And again, we produce a graph.
        pcolor(xx, yy, IterIndexMatrix)
        show(pcolor)
















# Here we define Two of the Function that we plot together
# We also define their parital derivatives.
# The functions take in either an array ora list as input.
def F(x):
    return x[0] ** 3 - 3 * x[0] * x[1] ** 2 - 1

# Works
def G(x):
    return 3 * x[0] ** 2 * x[1] - x[1] ** 3

# Works
def Fder1(x):
    return 3 * x[0] ** 2 - 3 * x[1] ** 2

# Works
def Fder2(x):
    return -6 * x[0] * x[1]

# Works
def Gder1(x):
    return 6 * x[0] * x[1]

# Works
def Gder2(x):
    return 3 * x[0] ** 2 - 3 * x[1] ** 2

# Here we define other two function to test out.

def H(x):
    return x[0]**3 - 3*x[0]*x[1]**2-2*x[0]-2

def I(x):
    return 3*x[0]**2*x[1]-x[1]**3-2*x[1]

def Hder1(x):
    return 3*x[0]**2-3*x[1]**2-2

def Hder2(x):
    return -6*x[0]*x[1]

def Ider1(x):
    return 6*x[0]*x[1]

def Ider2(x):
    return 3*x[0]**2-3*x[1]**2-2

# Even more functions to test!

def J(x):
    return x[0]**8 -28*x[0]**6*x[1]**2 +70*x[0]**4*x[1]**4 +15*x[0]**4 -28*x[0]**2*x[1]**6 -90*x[0]**2*x[1]**2 +15*x[1]**4 -16

def K(x):
    return 8*x[0]**7*x[1] -56*x[0]**5*x[1]**3 +56*x[0]**3*x[1]**5 +60*x[0]**3*x[1] -8*x[0]*x[1]**7 -60*x[0]*x[1]**3

def Jder1(x):
    return 8*x[0]**7 +280*x[0]**3*x[1]**4 +60*x[0]**3 -168*x[0]**5*x[1]**2 -56*x[0]*x[1]**6 -180*x[0]*x[1]**2

def Jder2(x):
    return 8*x[1]**7 +60*x[1]**3 +280*x[0]**4*x[1]**3 -168*x[0]**2*x[1]**5 -180*x[0]**2*x[1] -56*x[0]**6*x[1]

def Kder1(x):
    return 56*x[1]*x[0]**6 -280*x[1]**3*x[0]**4 +168*x[0]**5*x[0]**2 +180*x[1]*x[0]**2 -8*x[1]**7-60*x[1]**3

def Kder2(x):
    return 8*x[0]**7 - 168*x[0]**5*x[1]**2 +280*x[0]**3*x[1]**4 + 60*x[0]**3 -56*x[0]*x[1]**6 -180*x[0]*x[1]**2
#Test function and its derivative

Duck = Fractal2D(F, G, Fder1, Fder2, Gder1, Gder2)
Otter = Fractal2D(H, I, Hder1, Hder2, Ider1, Ider2)
Snake = Fractal2D(J, K, Jder1, Jder2, Kder1, Kder2)

#start = time.time()
#for i in range(10000):
#    Duck.NewtonBad([3,2], False)
#end = time.time()
#print(end - start)

#start = time.time()
#for i in range(10000):
#    Duck.NewtonGood([3,2], True)
#end = time.time()
#print(end - start)

#start = time.time()
#for i in range(10000):
#    Duck.NewtonSimplified([3,2], False)
#end = time.time()
#print(end - start)


#start = time.time()
#print(Duck.NewtonBad([1,5], True))
#print(Duck.NewtonGood([1,5], True))
#print(Duck.NewtonSimplified([1,5], True))
#print(Duck.NewtonSimplified([1,5], False))
#print(' ')
#print(Duck.NewZero([1,5], 1, True))
#print(Duck.NewZero([1,5], 2, True))
#print(Duck.NewZero([1,5], 3, True))
#print(' ')
#print(Duck.NewZero([1,5], 1, False))
#print(Duck.NewZero([1,5], 2, False))
#print(Duck.NewZero([1,5], 3, False))
#print(Duck.zeroes)


#end = time.time()
#print(end-start)

#start = time.time()
#print(Duck.NewtonBad([1,3]))
#end = time.time()
#print(end-start)

#start = time.time()
#print(Duck.NewtonSimplified([2,3]))
#end = time.time()
#print(end-start)
#print(Duck.NewtonMethod([123456,7645385]))
#print(Duck.NewZero([3,2]))
#Duck.NewZero([123456,7645385])
#Duck.NewZero([1,2])
#print(Duck.zeroes)
#print(Duck.NewZero([123456,7645385]))

Snake.PlotIter(500, 0, 10, 0, 10, False)
#Otter.PlotIter(100, 0, 10, 0, 10, False)
#Otter.PlotIter(500, 2, 3, 3, 4, False)


