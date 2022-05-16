--a. How many orders were shipped by Speedy Express in total? 54--
SELECT S.ShipperName AS Shipper, COUNT(OrderID) AS NumberOfOrder
FROM Orders O
LEFT JOIN Shippers S
WHERE O.ShipperID = S.ShipperID
GROUP BY Shipper;

--b. What is the last name of the employee with the most orders? Peacock--
SELECT E.LastName AS Employee_Last_Name, COUNT(OrderID) AS Number_Of_Orders
FROM Orders O
LEFT JOIN Employees E
WHERE O.EmployeeID = E.EmployeeID
GROUP BY Employee_Last_Name
ORDER BY Number_Of_Orders DESC
LIMIT 5;

--c. What product was ordered the most by customers in Germany? Gorgonzola Telino--
SELECT P.ProductName, C.Country, COUNT(O.OrderID) AS Number_Of_Orders
FROM Orders O
LEFT JOIN Customers C
ON O.CustomerID = C.CustomerID
LEFT JOIN OrderDetails D
ON O.OrderID = D.OrderID
LEFT JOIN Products P
ON D.ProductID = P.ProductID
WHERE C.Country = "Germany"
GROUP BY P.ProductName
ORDER BY Number_Of_Orders DESC
LIMIT 5;