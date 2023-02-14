--Create Table for Property Data
CREATE TABLE GoMarlins (
    policyID INTEGER,
    statecode CHAR,
    county CHAR,
    eq_site_limit REAL,
    hu_site_limit REAL,
    fl_site_limit REAL,
    fr_site_limit REAL,
    tiv_2011 REAL,
    tiv_2012 REAL,
    eq_site_deductible REAL,
    hu_site_deductible REAL,
    fl_site_deductible REAL,
    fr_site_deductible REAL,
    point_latitude REAL,
    point_longitude REAL,
    line CHAR,
    construction CHAR,
    point_granularity INTEGER
);
--Import in our csv data set
.mode csv
.import FL_insurance_sample.csv GoMarlins
--Finally print, list, average, and create a frequency table from our data
SELECT * FROM GoMarlins LIMIT 10;
SELECT DISTINCT county FROM GoMarlins;
SELECT AVG(tiv_2012 - tiv_2011) FROM GoMarlins;
SELECT construction, COUNT(*) FROM GoMarlins GROUP BY construction;

