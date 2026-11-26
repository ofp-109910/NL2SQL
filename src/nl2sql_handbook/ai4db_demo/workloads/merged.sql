-- 离线对照候选；仍须通过同样的数据一致性及性能门槛，不会被直接采用。
SELECT c.city, SUM(o.quantity * o.unit_price) AS sales,
       COUNT(*) AS order_count, SUM(o.quantity) AS units
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = '已完成'
GROUP BY c.city
ORDER BY sales DESC, c.city ASC NULLS LAST;
