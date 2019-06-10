DROP VIEW IF EXISTS q0, q1i, q1ii, q1iii, q1iv, q2i, q2ii, q2iii, q3i, q3ii, q3iii_helper, q3iii, q4i, salary_2016, salary_bounds, binbox_count, q4ii, q4iii, q4iv, q4v;

-- Question 0
CREATE VIEW q0(era)
AS
  -- SELECT 1 -- replace this line
  SELECT MAX(era)
  FROM pitching
;

-- Question 1i
CREATE VIEW q1i(namefirst, namelast, birthyear)
AS
  -- SELECT 1, 1, 1 -- replace this line
  SELECT namefirst, namelast, birthyear
  FROM people
  WHERE weight > 300
;

-- Question 1ii
CREATE VIEW q1ii(namefirst, namelast, birthyear)
AS
  --SELECT 1, 1, 1 -- replace this line
  SELECT namefirst, namelast, birthyear
  FROM people
  WHERE namefirst LIKE '% %'
;

-- Question 1iii
CREATE VIEW q1iii(birthyear, avgheight, count)
AS
  --SELECT 1, 1, 1 -- replace this line
  SELECT birthyear, AVG(height) AS avgheight, COUNT(*) AS count
  FROM people
  GROUP BY birthyear
  ORDER BY birthyear

;

-- Question 1iv
CREATE VIEW q1iv(birthyear, avgheight, count)
AS
  --SELECT 1, 1, 1 -- replace this line
  SELECT *
  FROM q1iii
  WHERE avgheight > 70
;

-- Question 2i
CREATE VIEW q2i(namefirst, namelast, playerid, yearid)
AS
  ---SELECT 1, 1, 1, 1 -- replace this line
  SELECT p.namefirst AS namefirst, p.namelast AS namelast, p.playerid AS playerid, h.yearid AS yearid
  FROM halloffame AS h, people AS p
  WHERE h.inducted = 'Y' AND h.playerid = p.playerid
  ORDER BY yearid DESC
;

-- Question 2ii
CREATE VIEW q2ii(namefirst, namelast, playerid, schoolid, yearid)
AS
  --SELECT 1, 1, 1, 1, 1 -- replace this line
  SELECT qprev.namefirst AS namefirst, qprev.namelast AS namelast, qprev.playerid AS playerid, cp.schoolid AS schoolid, qprev.yearid AS yearid
  FROM q2i AS qprev, collegeplaying AS cp, schools AS sch
  WHERE qprev.playerid = cp.playerid AND cp.schoolid = sch.schoolid AND sch.schoolstate = 'CA'
  ORDER BY yearid DESC, schoolid, playerid
;

-- Question 2iii
CREATE VIEW q2iii(playerid, namefirst, namelast, schoolid)
AS
  --SELECT 1, 1, 1, 1 -- replace this line
  SELECT qprev.playerid AS playerid, qprev.namefirst AS namefirst, qprev.namelast AS namelast, cp.schoolid AS schoolid
  FROM q2i AS qprev LEFT OUTER JOIN collegeplaying AS cp ON qprev.playerid = cp.playerid
  ORDER BY playerid DESC, schoolid
;

---------------------------------------------------------------------------------------------------

-- Question 3i
CREATE VIEW q3i(playerid, namefirst, namelast, yearid, slg)
AS
  --SELECT 1, 1, 1, 1, 1 -- replace this line
  SELECT b.playerid AS playerid, p.namefirst AS namefirst, p.namelast AS namelast, b.yearid AS yearid,
  CAST((1 * (b.h - (b.h2b + b.h3b + b.hr))) + (2 * b.h2b) + (3 * b.h3b) + (4 * b.hr) AS FLOAT) / b.ab AS slg
  FROM batting AS b, people AS p
  WHERE b.ab > 50 AND b.playerid = p.playerid
  ORDER BY slg DESC, yearid, playerid
  LIMIT 10
;


---------------------------------------------------------------------------------------------------
-- Question 3ii
CREATE VIEW q3ii(playerid, namefirst, namelast, lslg)
AS

  SELECT p.playerid AS playerid, p.namefirst AS namefirst, p.namelast AS namelast,
  SUM(CAST((1 * (b.h - (b.h2b + b.h3b + b.hr))) + (2 * b.h2b) + (3 * b.h3b) + (4 * b.hr) AS FLOAT))/ SUM(b.ab) AS lslg
  FROM batting AS b, people AS p
  WHERE b.playerid = p.playerid
  GROUP BY p.playerid
  HAVING SUM(b.ab) > 50
  ORDER BY lslg DESC, playerid
  LIMIT 10
;

---------------------------------------------------------------------------------------------------

CREATE VIEW q3iii_helper(playerid, namefirst, namelast, lslg)
AS
  SELECT p.playerid AS playerid, p.namefirst AS namefirst, p.namelast AS namelast,
  SUM(CAST((1 * (b.h - (b.h2b + b.h3b + b.hr))) + (2 * b.h2b) + (3 * b.h3b) + (4 * b.hr) AS FLOAT))/ SUM(b.ab) AS lslg
  FROM batting AS b, people AS p
  WHERE b.playerid = p.playerid
  GROUP BY p.playerid
  HAVING SUM(b.ab) > 50
  ORDER BY lslg DESC, playerid
;

---------------------------------------------------------------------------------------------------
-- Question 3iii
CREATE VIEW q3iii(namefirst, namelast, lslg)
AS
  --SELECT 1, 1, 8 -- replace this line
  SELECT q33help.namefirst AS namefirst, q33help.namelast AS namelast, q33help.lslg AS lslg
  FROM q3iii_helper AS q33help
  WHERE q33help.lslg > (
    SELECT q33help.lslg
    FROM q3iii_helper AS q33help
    WHERE q33help.playerid = 'mayswi01'
  )

;

---------------------------------------------------------------------------------------------------
-- Question 4i
CREATE VIEW q4i(yearid, min, max, avg, stddev)
AS
  --SELECT 1, 1, 1, 1, 1 -- replace this line
  SELECT s.yearid AS yearid, MIN(s.salary) AS min, MAX(s.salary) AS max, AVG(s.salary) AS avg,
  STDDEV(s.salary) AS stddev
  FROM salaries AS s
  GROUP BY s.yearid
  ORDER BY yearid
;

---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
CREATE VIEW salary_2016(salary)
AS
  SELECT s.salary AS salary
  FROM salaries AS s
  WHERE s.yearid = 2016
;
---------------------------------------------------------------------------------------------------
CREATE VIEW salary_bounds(min, max, width)
AS
  SELECT MIN(s2016.salary) AS min, MAX(s2016.salary) AS max, (MAX(s2016.salary) - MIN(s2016.salary)) / 10 AS width
  FROM salary_2016 AS s2016
;

---------------------------------------------------------------------------------------------------
CREATE VIEW binbox_count(binid, count)
AS
  SELECT width_bucket(s2016.salary, bounds.min, bounds.max + 1, 10) - 1 AS binid, COUNT(*) AS count
  FROM salary_2016 AS s2016, salary_bounds AS bounds
  GROUP BY binid
  ORDER BY binid
;

---------------------------------------------------------------------------------------------------
-- Question 4ii
CREATE VIEW q4ii(binid, low, high, count)
AS
  --SELECT 1, 1, 1, 1 -- replace this line
  SELECT bbc.binid AS binid, bounds.min + (bounds.width * bbc.binid) AS low, bounds.min + (bounds.width * (bbc.binid + 1)) AS high, bbc.count AS count
  FROM binbox_count AS bbc, salary_bounds AS bounds
  ORDER BY binid
;


---------------------------------------------------------------------------------------------------
-- Question 4iii
CREATE VIEW q4iii(yearid, mindiff, maxdiff, avgdiff)
AS
  --SELECT 1, 1, 1, 1 -- replace this line
  SELECT thisyear.yearid AS yearid, (thisyear.min - preyear.min) AS mindiff, (thisyear.max - preyear.max) AS maxdiff, (thisyear.avg - preyear.avg) AS avgdiff
  FROM q4i AS thisyear INNER JOIN q4i AS preyear ON preyear.yearid = thisyear.yearid - 1
  ORDER BY yearid
;


---------------------------------------------------------------------------------------------------
-- Question 4iv
CREATE VIEW q4iv(playerid, namefirst, namelast, salary, yearid)
AS
  --SELECT 1, 1, 1, 1, 1 -- replace this line
  SELECT s.playerid AS playerid, p.namefirst AS namefirst, p.namelast AS namelast, s.salary AS salary, s.yearid AS yearid
  FROM people AS p, salaries AS s
  WHERE p.playerid = s.playerid AND
        s.yearid IN (2000, 2001) AND
        (s.yearid, s.salary) IN (
          SELECT s.yearid, MAX(s.salary)
          FROM salaries AS s
          GROUP BY s.yearid
        )

;

---------------------------------------------------------------------------------------------------
-- Question 4v
CREATE VIEW q4v(team, diffAvg)
AS
  --SELECT 1, 1 -- replace this line
  SELECT asf.teamid AS team, MAX(s.salary) - MIN(s.salary) AS diffAvg
  FROM allstarfull AS asf, salaries AS s
  WHERE asf.playerid = s.playerid AND asf.teamid = s.teamid AND asf.yearid = s.yearid AND asf.yearid = 2016
  GROUP BY team
  ORDER BY team

;
