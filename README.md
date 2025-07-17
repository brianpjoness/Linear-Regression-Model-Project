# This is README

Creating a linear regression model from scratch
The goal is to minimize the error 


Error Function: Mean Squared Error

E = (1/n) * sum (i -> n) (yi - (mxi +b))^2
Average value of the difference between actual and predicted values squared


How we will minimize E:
y = mx + b
We can change m and b, so we take partial derivative with respect to m and b
Allows you to find the direction of the steepest ascent (with partial derivative) so you take opposite direction

m = m - L (dE/dm)
b = b - L (dE/db)

L is the learning rate that determines how quickly it will come to a solution, lower learning rate is less error
