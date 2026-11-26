-- 三个独立聚合重复扫描同一组表；DuckDB 自身的优化器始终开启。
WITH sales AS (
    SELECT c.city, SUM(o.quantity * o.unit_price) AS sales
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = '已完成' GROUP BY c.city
), counts AS (
    SELECT c.city, COUNT(*) AS order_count
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = '已完成' GROUP BY c.city
), units AS (
    SELECT c.city, SUM(o.quantity) AS units
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = '已完成' GROUP BY c.city
)
SELECT s.city, s.sales, n.order_count, u.units
FROM sales s JOIN counts n ON s.city IS NOT DISTINCT FROM n.city
JOIN units u ON s.city IS NOT DISTINCT FROM u.city
ORDER BY s.sales DESC, s.city ASC NULLS LAST;
