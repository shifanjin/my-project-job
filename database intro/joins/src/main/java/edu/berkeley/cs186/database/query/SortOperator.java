package edu.berkeley.cs186.database.query;

import edu.berkeley.cs186.database.Database;
import edu.berkeley.cs186.database.DatabaseException;
import edu.berkeley.cs186.database.databox.DataBox;
import edu.berkeley.cs186.database.table.Record;
import edu.berkeley.cs186.database.table.Schema;
import edu.berkeley.cs186.database.common.Pair;
import edu.berkeley.cs186.database.io.Page;
import edu.berkeley.cs186.database.common.BacktrackingIterator;

import java.util.*;

public class SortOperator {
    private Database.Transaction transaction;
    private String tableName;
    private Comparator<Record> comparator;
    private Schema operatorSchema;
    private int numBuffers;
    private String sortedTableName = null;

    public SortOperator(Database.Transaction transaction, String tableName,
                        Comparator<Record> comparator) throws DatabaseException, QueryPlanException {
        this.transaction = transaction;
        this.tableName = tableName;
        this.comparator = comparator;
        this.operatorSchema = this.computeSchema();
        this.numBuffers = this.transaction.getNumMemoryPages();
    }

    public Schema computeSchema() throws QueryPlanException {
        try {
            return this.transaction.getFullyQualifiedSchema(this.tableName);
        } catch (DatabaseException de) {
            throw new QueryPlanException(de);
        }
    }

    public class Run {
        String tempTableName;

        public Run() throws DatabaseException {
            this.tempTableName = SortOperator.this.transaction.createTempTable(
                                     SortOperator.this.operatorSchema);
        }

        public void addRecord(List<DataBox> values) throws DatabaseException {
            SortOperator.this.transaction.addRecord(this.tempTableName, values);
        }

        public void addRecords(List<Record> records) throws DatabaseException {
            for (Record r : records) {
                this.addRecord(r.getValues());
            }
        }

        public Iterator<Record> iterator() throws DatabaseException {
            return SortOperator.this.transaction.getRecordIterator(this.tempTableName);
        }

        public String tableName() {
            return this.tempTableName;
        }
    }

    /**
     * Returns a NEW run that is the sorted version of the input run.
     * Can do an in memory sort over all the records in this run
     * using one of Java's built-in sorting methods.
     * Note: Don't worry about modifying the original run.
     * Returning a new run would bring one extra page in memory beyond the
     * size of the buffer, but it is done this way for ease.
     */
    public Run sortRun(Run run) throws DatabaseException {
//        throw new UnsupportedOperationException("TODO(hw3): implement");
        Iterator<Record> iter = run.iterator();
        ArrayList<Record> addtheserecords = new ArrayList<>();
        Run sortedruns = this.createRun();

        while (iter.hasNext()) {
            addtheserecords.add(iter.next());
        }

        addtheserecords.sort(this.comparator);  //Now presortlist is sorted!

        // Add the sorted list of records to the created sortedruns
        sortedruns.addRecords(addtheserecords);
        return sortedruns;
    }

    /**
     * Given a list of sorted runs, returns a new run that is the result
     * of merging the input runs. You should use a Priority Queue (java.util.PriorityQueue)
     * to determine which record should be should be added to the output run next.
     * It is recommended that your Priority Queue hold Pair<Record, Integer> objects
     * where a Pair (r, i) is the Record r with the smallest value you are
     * sorting on currently unmerged from run i.
     */
    public Run mergeSortedRuns(List<Run> runs) throws DatabaseException {
//        throw new UnsupportedOperationException("TODO(hw3): implement");

        Run mergedruns = this.createRun();
        int runs_size = runs.size();
        Queue<Pair<Record, Integer>> pq = new PriorityQueue<>(new RecordPairComparator());

        for (int i = 0; i < runs_size; i++) {

            Run ith_run = runs.get(i);
            Iterator<Record> ith_iter = ith_run.iterator();

            while (ith_iter.hasNext()) {

                Pair<Record, Integer> ith_jth_pair= new Pair<Record, Integer> (ith_iter.next(), i);
                pq.add(ith_jth_pair);

            }
        }


        while (!pq.isEmpty()) {
            //poll() returns and removes the element at the front the container/queue
            Pair<Record, Integer> thispair = pq.poll();
            mergedruns.addRecord(thispair.getFirst().getValues());
        }

        return mergedruns;
    }

    /**
     * Given a list of N sorted runs, returns a list of
     * sorted runs that is the result of merging (numBuffers - 1)
     * of the input runs at a time.
     */
    public List<Run> mergePass(List<Run> runs) throws DatabaseException {
//        throw new UnsupportedOperationException("TODO(hw3): implement");
        ArrayList<Run> mergepass = new ArrayList<>();
        int runs_size = runs.size();

        //merging (numBuffers - 1) at a time
        int mergethismuch = this.numBuffers - 1;

        for (int i = 0; i < runs_size; i += mergethismuch) {

            // get the sublist
            List<Run> list = runs.subList(i, i + mergethismuch);
            mergepass.add(mergeSortedRuns(list));

        }
        return mergepass;
    }

    /**
     * Does an external merge sort on the table with name tableName
     * using numBuffers.
     * Returns the name of the table that backs the final run.
     */
    public String sort() throws DatabaseException {
//        throw new UnsupportedOperationException("TODO(hw3): implement");

        String thetablename;
        List<Run> sortedruns = new ArrayList<>();
        // FIRST GET THE PAGE ITER
        BacktrackingIterator<Page> pageiter = this.transaction.getPageIterator(this.tableName);
        // skip header
        pageiter.next();

        while (pageiter.hasNext()) {

            Run presortedrun = this.createRun();

            // GET THE RECORD ITER
            BacktrackingIterator<Record> recorditer = this.transaction.getBlockIterator(this.tableName, pageiter, this.numBuffers);

            while (recorditer.hasNext()) {

                Record addthisrecord = recorditer.next();
                presortedrun.addRecord(addthisrecord.getValues());

            }

            // Remember to sort the runs!
            sortedruns.add(sortRun(presortedrun));
        }


        while (sortedruns.size() > 1) {

            sortedruns = mergePass(sortedruns);

        }

        Run therun = sortedruns.get(0);
        // get the table name!
        thetablename = therun.tableName();

        return thetablename;

    }

    public Iterator<Record> iterator() throws DatabaseException {
        if (sortedTableName == null) {
            sortedTableName = sort();
        }
        return this.transaction.getRecordIterator(sortedTableName);
    }

    private class RecordPairComparator implements Comparator<Pair<Record, Integer>> {

        public int compare(Pair<Record, Integer> o1, Pair<Record, Integer> o2) {
            return SortOperator.this.comparator.compare(o1.getFirst(), o2.getFirst());

        }
    }

    public Run createRun() throws DatabaseException {
        return new Run();
    }
}

