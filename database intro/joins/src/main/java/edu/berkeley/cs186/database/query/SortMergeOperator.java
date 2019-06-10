package edu.berkeley.cs186.database.query;

import java.util.*;

import edu.berkeley.cs186.database.Database;
import edu.berkeley.cs186.database.DatabaseException;
import edu.berkeley.cs186.database.databox.DataBox;
import edu.berkeley.cs186.database.table.Record;
import edu.berkeley.cs186.database.table.RecordIterator;

public class SortMergeOperator extends JoinOperator {
    public SortMergeOperator(QueryOperator leftSource,
                             QueryOperator rightSource,
                             String leftColumnName,
                             String rightColumnName,
                             Database.Transaction transaction) throws QueryPlanException, DatabaseException {
        super(leftSource, rightSource, leftColumnName, rightColumnName, transaction, JoinType.SORTMERGE);

        // for HW4
        this.stats = this.estimateStats();
        this.cost = this.estimateIOCost();
    }

    public Iterator<Record> iterator() throws QueryPlanException, DatabaseException {
        return new SortMergeIterator();
    }

    public int estimateIOCost() throws QueryPlanException {
        //does nothing
        return 0;
    }

    /**
     * An implementation of Iterator that provides an iterator interface for this operator.
     *
     * Before proceeding, you should read and understand SNLJOperator.java
     *    You can find it in the same directory as this file.
     *
     * Word of advice: try to decompose the problem into distinguishable sub-problems.
     *    This means you'll probably want to add more methods than those given (Once again,
     *    SNLJOperator.java might be a useful reference).
     *
     */
    private class SortMergeIterator extends JoinIterator {
        /**
        * Some member variables are provided for guidance, but there are many possible solutions.
        * You should implement the solution that's best for you, using any member variables you need.
        * You're free to use these member variables, but you're not obligated to.
        */

        private String leftTableName;
        private String rightTableName;
        private RecordIterator leftIterator;
        private RecordIterator rightIterator;
        private Record leftRecord;
        private Record nextRecord;
        private Record rightRecord;
        private boolean marked;
        private LR_RecordComparator LR_comp;



        public SortMergeIterator() throws QueryPlanException, DatabaseException {
            super();
//            throw new UnsupportedOperationException("TODO(hw3): implement");
            SortOperator leftsortop = new SortOperator(SortMergeOperator.this.getTransaction(),
                    this.getLeftTableName(), new LeftRecordComparator());
            SortOperator rightsortop = new SortOperator(SortMergeOperator.this.getTransaction(),
                    this.getRightTableName(), new RightRecordComparator());

            this.leftTableName = leftsortop.sort();
            this.rightTableName = rightsortop.sort();

            this.leftIterator = SortMergeOperator.this.getRecordIterator(leftTableName);
            this.rightIterator = SortMergeOperator.this.getRecordIterator(rightTableName);


            if (this.leftIterator.hasNext()) {
                this.leftRecord = this.leftIterator.next();
            } else {
                this.leftRecord = null;
            }

            if (this.rightIterator.hasNext()) {
                this.rightRecord = this.rightIterator.next();
            } else {
                this.rightRecord = null;
            }


            this.nextRecord = null;
            this.marked = false;
            this.LR_comp = new LR_RecordComparator();

            try {
                fetchNextRecord();
            } catch (DatabaseException e) {
                this.nextRecord = null;
            }
        }

        private void fetchNextRecord() throws DatabaseException {

            if (this.leftRecord == null) throw new DatabaseException("No new record to fetch");
            this.nextRecord = null;
            do {

                if (this.marked == false) {

                    if (this.leftRecord == null) {
                        throw new DatabaseException("No new record to fetch");
                    }

                    while (this.leftRecord != null
                            && this.LR_comp.compare(leftRecord, rightRecord) < 0) {
                        if (this.leftIterator.hasNext()) {
                            this.leftRecord = this.leftIterator.next();
                        } else {
                            this.leftRecord = null;
                        }
                    }

                    while (this.rightRecord != null
                            && this.LR_comp.compare(leftRecord, rightRecord) > 0) {
                        if (this.rightIterator.hasNext()) {
                            this.rightRecord = this.rightIterator.next();
                        } else {
                            this.rightRecord = null;
                        }
                    }

                    this.rightIterator.mark();
                    this.marked = true;
                }

                if (this.leftRecord != null && this.rightRecord != null
                        && LR_comp.compare(leftRecord, rightRecord) == 0) {
                    List<DataBox> leftlist = new ArrayList<>(this.leftRecord.getValues());
                    List<DataBox> rightlist = new ArrayList<>(this.rightRecord.getValues());
                    // PUT TOGETHER
                    leftlist.addAll(rightlist);

                    if (this.rightIterator.hasNext()) {
                        this.rightRecord = this.rightIterator.next();
                    } else {
                        this.rightRecord = null;
                    }

                    this.nextRecord = new Record(leftlist);


                } else {

                    this.rightIterator.reset();

                    if (this.leftIterator.hasNext()) {
                        this.leftRecord = this.leftIterator.next();
                    } else {
                        this.leftRecord = null;
                    }

                    if (this.rightIterator.hasNext()) {
                        this.rightRecord = this.rightIterator.next();
                    } else {
                        this.rightRecord = null;
                    }

                    this.marked = false;
                }

            } while (!hasNext());
        }


        /**
         * Checks if there are more record(s) to yield
         *
         * @return true if this iterator has another record to yield, otherwise false
         */
        public boolean hasNext() {
//            throw new UnsupportedOperationException("TODO(hw3): implement");
            return this.nextRecord != null;
        }


        /**
         * Yields the next record of this iterator.
         *
         * @return the next Record
         * @throws NoSuchElementException if there are no more Records to yield
         */
        public Record next() {
//            throw new UnsupportedOperationException("TODO(hw3): implement");
            if (!this.hasNext()) {
                throw new NoSuchElementException();
            }

            Record nextRecord = this.nextRecord;

            try {
                this.fetchNextRecord();
            } catch (DatabaseException e) {
                this.nextRecord = null;
            }
            return nextRecord;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        private class LeftRecordComparator implements Comparator<Record> {
            public int compare(Record o1, Record o2) {
                return o1.getValues().get(SortMergeOperator.this.getLeftColumnIndex()).compareTo(
                           o2.getValues().get(SortMergeOperator.this.getLeftColumnIndex()));
            }
        }

        private class RightRecordComparator implements Comparator<Record> {
            public int compare(Record o1, Record o2) {
                return o1.getValues().get(SortMergeOperator.this.getRightColumnIndex()).compareTo(
                           o2.getValues().get(SortMergeOperator.this.getRightColumnIndex()));
            }
        }

        /**
        * Left-Right Record comparator
        * o1 : leftRecord
        * o2: rightRecord
        */
        private class LR_RecordComparator implements Comparator<Record> {
            public int compare(Record o1, Record o2) {
                return o1.getValues().get(SortMergeOperator.this.getLeftColumnIndex()).compareTo(
                           o2.getValues().get(SortMergeOperator.this.getRightColumnIndex()));
            }
        }
    }
}
