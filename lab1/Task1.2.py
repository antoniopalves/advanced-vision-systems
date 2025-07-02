from matplotlib . patches import Rectangle # add at the top of the file
import matplotlib . pyplot as plt
I = plt.imread (r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg")
fig , ax = plt . subplots (1) # instead of plt . figure (1)
plt . imshow (I) # add image

x = [ 100 , 150 , 200 , 250]
y = [ 50 , 100 , 150 , 200]
plt.plot (x ,y ,"r.", markersize =10)

rect = Rectangle ((50 ,50) ,50 ,100 , fill = False , ec ="r"); # ec - edge colour
ax . add_patch ( rect ) # display

plt . title ("Mandril nº2") # add title
plt . axis ("off") # disable display of the coordinate system
plt . show () # display
plt.imsave (r"C:\Users\anton\Documents\Erasmus\AVS\mandrill2.jpg",I)


