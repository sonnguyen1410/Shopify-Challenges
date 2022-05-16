# Shopify Fall 2022 Data Science Intern Challenge
## Question 2:
### Please use queries to answer the following questions. Paste your queries along with your final numerical answers below.
a. How many orders were shipped by Speedy Express in total?

Answer: 54

Queries:
```sql
SELECT S.ShipperName AS Shipper, COUNT(OrderID) AS NumberOfOrder
FROM Orders O
LEFT JOIN Shippers S
WHERE O.ShipperID = S.ShipperID
GROUP BY Shipper;
```
Result:
```
Shipper	NumberOfOrder
Federal Shipping	68
Speedy Express	54
United Package	74
```

b. What is the last name of the employee with the most orders?

Answer: Peacock

Queries:
```sql
SELECT E.LastName AS Employee_Last_Name, COUNT(OrderID) AS Number_Of_Orders
FROM Orders O
LEFT JOIN Employees E
WHERE O.EmployeeID = E.EmployeeID
GROUP BY Employee_Last_Name
ORDER BY Number_Of_Orders DESC
LIMIT 5;
```

Result:
```
Employee_Last_Name	Number_Of_Orders
Peacock	40
Leverling	31
Davolio	29
Callahan	27
Fuller	20
```

c. What product was ordered the most by customers in Germany?

Answer: Gorgonzola Telino

Queries:
```sql
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
```

Result:
```
ProductName	Country	Number_Of_Orders
Gorgonzola Telino	Germany	5
Lakkalikööri	Germany	4
Boston Crab Meat	Germany	4
Tunnbröd	Germany	3
Mozzarella di Giovanni	Germany	3
```