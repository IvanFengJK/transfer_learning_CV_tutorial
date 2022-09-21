import matplotlib.pyplot as plt

def visualiseList(train, val):
        plt.figure()
        x1 = list(range(len(train)))
        # corresponding y axis values
        y1 = train

        # plotting the points 
        plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 0.5,
                marker='o', markerfacecolor='blue', markersize=5, label = "Traing")

        x2 = list(range(len(val)))
        # corresponding y axis values
        y2 = val

        # plotting the points 
        plt.plot(x2, y2, color='red', linestyle='solid', linewidth = 0.5, 
        markerfacecolor='green', markersize=5, label = "Validation")

        # setting x and y axis range
        plt.xlim(0,len(train))
        plt.ylim(0,1)
        #plt.ylim(min(min(y1),min(y2))-0.1,max(max(y1),max(y2))+0.1)

        # naming the x axis
        plt.xlabel('Epoch')
        # naming the y axis
        plt.ylabel('Accuracy')

        # giving a title to my graph
        plt.title('Trend of accuracy over epoches')
        plt.legend()

        # function to show the plot
        plt.show()