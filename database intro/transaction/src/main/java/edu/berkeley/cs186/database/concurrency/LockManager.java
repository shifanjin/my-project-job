package edu.berkeley.cs186.database.concurrency;

import edu.berkeley.cs186.database.BaseTransaction;

import java.util.*;
import java.util.stream.Collectors;

/**
 * LockManager maintains the bookkeeping for what transactions have
 * what locks on what resources. The lock manager should generally **not**
 * be used directly: instead, code should call methods of LockContext to
 * acquire/release/promote/escalate locks.
 *
 * The LockManager is primarily concerned with the mappings between
 * transactions, resources, and locks, and does not concern itself with
 * multiple levels of granularity (you can and should treat ResourceName
 * as a generic Object, rather than as an object encapsulating levels of
 * granularity, in this class).
 *
 * It follows that LockManager should allow **all**
 * requests that are valid from the perspective of treating every resource
 * as independent objects, even if they would be invalid from a
 * multigranularity locking perspective. For example, if LockManager#acquire
 * is called asking for an X lock on Table A, and the transaction has no
 * locks at the time, the request is considered valid (because the only problem
 * with such a request would be that the transaction does not have the appropriate
 * intent locks, but that is a multigranularity concern).
 *
 * Each resource the lock manager manages has its own queue of LockRequest objects
 * representing a request to acquire (or promote/acquire-and-release) a lock that
 * could not be satisfied at the time. This queue should be processed every time
 * a lock on that resource gets released, starting from the first request, and going
 * in order until a request cannot be satisfied. Requests taken off the queue should
 * be treated as if that transaction had made the request right after the resource was
 * released in absence of a queue (i.e. removing a request by T1 to acquire X(db) should
 * be treated as if T1 had just requested X(db) and there were no queue on db: T1 should
 * be given the X lock on db, and put in an unblocked state via BaseTransaction#unblock).
 *
 * This does mean that in the case of:
 *    queue: S(A) X(A) S(A)
 * only the first request should be removed from the queue when the queue is processed.
 */
public class LockManager {
    // transactionLocks is a mapping from transaction number to a list of lock
    // objects held by that transaction.
    private Map<Long, List<Lock>> transactionLocks = new HashMap<>();
    // resourceEntries is a mapping from resource names to a ResourceEntry
    // object, which contains a list of Locks on the object, as well as a
    // queue for requests on that resource.
    private Map<ResourceName, ResourceEntry> resourceEntries = new HashMap<>();

    // A ResourceEntry contains the list of locks on a resource, as well as
    // the queue for requests for locks on the resource.
    private class ResourceEntry {
        // List of currently granted locks on the resource.
        List<Lock> locks = new ArrayList<>();
        // Queue for yet-to-be-satisfied lock requests on this resource.
        Deque<LockRequest> waitingQueue = new ArrayDeque<>();

        // You may add helper methods here if you wish
    }

    // You should not modify or use this directly.
    protected Map<Object, LockContext> contexts = new HashMap<>();

    /**
     * Helper method to fetch the resourceEntry corresponding to NAME.
     * Inserts a new (empty) resourceEntry into the map if no entry exists yet.
     */
    private ResourceEntry getResourceEntry(ResourceName name) {
        resourceEntries.putIfAbsent(name, new ResourceEntry());
        return resourceEntries.get(name);
    }

    // You may add helper methods here if you wish

    /**
     * Acquire a LOCKTYPE lock on NAME, for transaction TRANSACTION, and releases all locks
     * in RELEASELOCKS after acquiring the lock, in one atomic action.
     *
     * Error checking must be done before any locks are acquired or released. If the new lock
     * is not compatible with another transaction's lock on the resource, the transaction is
     * blocked and the request is placed at the **front** of ITEM's queue.
     *
     * Locks in RELEASELOCKS should be released only after the requested lock has been acquired.
     * The corresponding queues should be processed.
     *
     * An acquire-and-release that releases an old lock on NAME **should not** change the
     * acquisition time of the lock on NAME, i.e.
     * if a transaction acquired locks in the order: S(A), X(B), acquire X(A) and release S(A), the
     * lock on A is considered to have been acquired before the lock on B.
     *
     * @throws DuplicateLockRequestException if a lock on NAME is held by TRANSACTION and
     * isn't being released
     * @throws NoLockHeldException if no lock on a name in RELEASELOCKS is held by TRANSACTION
     */
    public void acquireAndRelease(BaseTransaction transaction, ResourceName name,
                                  LockType lockType, List<ResourceName> releaseLocks)
            throws DuplicateLockRequestException, NoLockHeldException {
        // You may modify any part of this method. You are not required to keep all your
        // code within the given synchronized block -- in fact,
        // you will have to write some code outside the synchronized block to avoid locking up
        // the entire lock manager when a transaction is blocked. You are also allowed to
        // move the synchronized block elsewhere if you wish.


        Long transNum = transaction.getTransNum();
        Lock newlock = new Lock(name, lockType, transNum);
//            ResourceEntry curRE = getResourceEntry(name);
        boolean hasSameTN = false;
        boolean lockheld = false;
        boolean isblocked = false;
        boolean compatible = false;


        synchronized (this) {
//            throw new UnsupportedOperationException("TODO(hw5_part1): implement");

//            Long transNum = transaction.getTransNum();
//            Lock newlock = new Lock(name, lockType, transNum);
////            ResourceEntry curRE = getResourceEntry(name);
//            boolean hasSameTN = false;
//            boolean lockheld = false;
//            boolean isblocked = false;
//            boolean compatible = false;

            ////////////////////////////////////////////////////////////////// ACQUIRE

            for (Map.Entry<Long, List<Lock>> eachTL : this.transactionLocks.entrySet()) {
                Long curtrans = eachTL.getKey();
                List<Lock> thisTransLocks = eachTL.getValue();

                // Lock on name is already held by transaction >> check if it's released or not!
                if (curtrans == transNum) {
                    hasSameTN = true;
                    for (int i = 0; i < thisTransLocks.size(); i++) {
                        if (thisTransLocks.get(i).name.equals(name)) {
                            // And the lock is not yet released >> throw exception!
                            if (releaseLocks.contains(name) == false) {
                                throw new DuplicateLockRequestException("The lock" + name + "is already held by" + transaction);
                            }
                        }
                    }

                } else {
                    for (int i = 0; i < thisTransLocks.size(); i++) {
                        if (thisTransLocks.get(i).name.equals(name)) {
                            compatible = LockType.compatible(thisTransLocks.get(i).lockType, lockType);
                            if (compatible == false) {
                                // block the transaction because the requested lock type is not compatible with other transactions
                                transaction.block();
                                LockRequest newLR = new LockRequest(transaction, newlock);
                                // get the ResourceEntry in order to add the new LockRequest to the target waitingQueue
                                this.getResourceEntry(name).waitingQueue.addLast(newLR);
                                isblocked= true;
                            }
                        }
                    }
                }
            }


            // add the newlock to our transactionLocks
            if (isblocked == false) {

                if (hasSameTN == true) {
                    this.transactionLocks.get(transNum).add(newlock);
                    this.getResourceEntry(name).locks.add(newlock);
                } else {
                    List<Lock> newlocklist = new ArrayList<>();
                    newlocklist.add(newlock);
                    this.transactionLocks.put(transNum, newlocklist);
                    this.getResourceEntry(name).locks.add(newlock);
                }

            }


            //////////////////////////////////////////////////////////////////RELEASE
            for (int i = 0; i < releaseLocks.size(); i++) {
                if (this.transactionLocks.containsKey(transNum)) {
//                    boolean lockFound = false;
                    List<Lock> transcurlocks = getLocks(transaction);
                    for (int lk = 0; lk < transcurlocks.size(); lk++) {
                        if (transcurlocks.get(lk).name == releaseLocks.get(i)) {
                            lockheld = true;
                            if (transaction.getBlocked() == false) {
                                this.transactionLocks.get(transNum).remove(transcurlocks.get(lk));
                                this.getResourceEntry(releaseLocks.get(i)).locks.remove(transcurlocks.get(lk));
                            }
                        }
                    }
                    if (lockheld == false) {
                        throw new NoLockHeldException("No lock held exception");
                    }
                } else {
                    throw new NoLockHeldException("No lock held exception");
                }
            }


            //////////////////////////////////////////////////////////////////

            Iterator iterator = this.getResourceEntry(name).waitingQueue.iterator();
            while (iterator.hasNext()) {
                LockRequest lockRequest = (LockRequest) iterator.next();
//                boolean compatible = true;
                boolean promotion = false;
                for (Map.Entry<Long, List<Lock>> eachTL : this.transactionLocks.entrySet()) {
                    Long tlnum = eachTL.getKey();
                    List<Lock> thelocks = eachTL.getValue();
                    for (int i = 0; i < thelocks.size(); i++) {
                        Lock curlocki = thelocks.get(i);
                        if (curlocki.name.equals(lockRequest.lock.name)) {
                            if (tlnum != lockRequest.transaction.getTransNum()) {
                                if (LockType.compatible(curlocki.lockType, lockRequest.lock.lockType) == false) {
                                    compatible = false;
                                }
                            } else {
                                promotion = true;
                            }
                        }
                    }
                }

                if (compatible == true) {
                    lockRequest.transaction.unblock();
                    if (promotion == true) {
                        promote(lockRequest.transaction, lockRequest.lock.name, lockRequest.lock.lockType);
                    } else {
                        acquire(lockRequest.transaction, lockRequest.lock.name, lockRequest.lock.lockType);
                    }
                    iterator.remove();
                }
            }

        }
    }

    /**
     * Acquire a LOCKTYPE lock on NAME, for transaction TRANSACTION.
     *
     * Error checking must be done before the lock is acquired. If the new lock
     * is not compatible with another transaction's lock on the resource, or if there are
     * other transaction in queue for the resource, the transaction is
     * blocked and the request is placed at the **back** of NAME's queue.
     *
     * @throws DuplicateLockRequestException if a lock on NAME is held by
     * TRANSACTION
     */
    public void acquire(BaseTransaction transaction, ResourceName name,
                        LockType lockType) throws DuplicateLockRequestException {
        // You may modify any part of this method. You are not required to keep all your
        // code within the given synchronized block -- in fact,
        // you will have to write some code outside the synchronized block to avoid locking up
        // the entire lock manager when a transaction is blocked. You are also allowed to
        // move the synchronized block elsewhere if you wish.

        Long transNum = transaction.getTransNum();
        Lock newlock = new Lock(name, lockType, transNum);
        boolean sameTransID = false;
        boolean blocked = false;

        synchronized (this) {
//            throw new UnsupportedOperationException("TODO(hw5_part1): implement");
//            Lock newlock = new Lock(name, lockType, transaction.getTransNum());
//            boolean sameTransID = false;
//            boolean blocked = false;
            for (Map.Entry<Long, List<Lock>> eachTL : this.transactionLocks.entrySet()) {
                Long thistrannum = eachTL.getKey();
                List<Lock> thelocks = eachTL.getValue();
                if (thistrannum == transNum) {
                    sameTransID = true;
                    for (int i = 0; i < thelocks.size(); i++) {
                        Lock curlock = thelocks.get(i);
                        if (curlock.name.equals(name)) {
                            throw new DuplicateLockRequestException("Duplicate lock request exception");
                        }
                    }

                } else {
                    for (int i = 0; i < thelocks.size(); i++) {
                        Lock curlock = thelocks.get(i);
                        if (curlock.name.equals(name)) {
                            if (LockType.compatible(thelocks.get(i).lockType, lockType) == false || this.getResourceEntry(name).waitingQueue == null) {
                                transaction.block();
                                LockRequest newRequest = new LockRequest(transaction, newlock);
                                getResourceEntry(name).waitingQueue.addLast(newRequest);
                                blocked = true;
                            }
                        }
                    }
                }
            }

            if (blocked == false) {
                if (sameTransID) {
                    this.transactionLocks.get(transNum).add(newlock);
                    this.getResourceEntry(name).locks.add(newlock);
                } else {
                    List<Lock> newlocklist = new ArrayList<>();
                    newlocklist.add(newlock);
                    this.transactionLocks.put(transNum, newlocklist);
                    this.getResourceEntry(name).locks.add(newlock);

                }
            } else {
                transaction.block();
            }
        }
    }

    /**
     * Release TRANSACTION's lock on NAME.
     *
     * Error checking must be done before the lock is released.
     *
     * NAME's queue should be processed after this call. If any requests in
     * the queue have locks to be released, those should be released, and the
     * corresponding queues also processed.
     *
     * @throws NoLockHeldException if no lock on NAME is held by TRANSACTION
     */
    public void release(BaseTransaction transaction, ResourceName name)
            throws NoLockHeldException {
        // You may modify any part of this method.

        Long transNum = transaction.getTransNum();
        boolean lockOnName = false;

        synchronized (this) {
//            throw new UnsupportedOperationException("TODO(hw5_part1): implement");
            if (this.transactionLocks.containsKey(transNum)) {
//                boolean lockOnName = false;
                List<Lock> thelocks = this.transactionLocks.get(transNum);
                for (int i = 0; i < thelocks.size(); i++) {
                    Lock curlock = thelocks.get(i);
                    if (curlock.name.equals(name)) {
                        lockOnName = true;
                        if (transaction.getBlocked() == false) {
                            this.transactionLocks.get(transNum).remove(curlock);
                            this.getResourceEntry(name).locks.remove(curlock);
                        }
                    }
                }

                if (!lockOnName) {
                    throw new NoLockHeldException("No lock held exception");
                }
            } else {
                throw new NoLockHeldException("No lock held exception");
            }

            Iterator iterator = getResourceEntry(name).waitingQueue.iterator();
            while (iterator.hasNext()) {
                LockRequest lockRequest = (LockRequest) iterator.next();
                boolean compatible = true;
                boolean promotion = false;
                for (Map.Entry<Long, List<Lock>> eachTL : this.transactionLocks.entrySet()) {
                    Long tlnum = eachTL.getKey();
                    List<Lock> thelocks = eachTL.getValue();

                    for (int i = 0; i < thelocks.size(); i++) {

                        Lock curlock = thelocks.get(i);
                        if (curlock.name.equals(lockRequest.lock.name)) {
                            if (tlnum != lockRequest.transaction.getTransNum()) {
                                if (LockType.compatible(curlock.lockType, lockRequest.lock.lockType) == false) {
                                    compatible = false;
                                }
                            } else {
                                promotion = true;
                            }
                        }
                    }
                }

                if (compatible) {
                    lockRequest.transaction.unblock();
                    if (promotion) {
                        promote(lockRequest.transaction, lockRequest.lock.name, lockRequest.lock.lockType);
                    } else {
                        acquire(lockRequest.transaction, lockRequest.lock.name, lockRequest.lock.lockType);
                    }
                    iterator.remove();
                }
            }
        }
    }

    /**
     * Promote TRANSACTION's lock on NAME to NEWLOCKTYPE (i.e. change TRANSACTION's lock
     * on NAME from the current lock type to NEWLOCKTYPE, which must be strictly more
     * permissive).
     *
     * Error checking must be done before any locks are changed. If the new lock
     * is not compatible with another transaction's lock on the resource, the transaction is
     * blocked and the request is placed at the **front** of ITEM's queue.
     *
     * A lock promotion **should not** change the acquisition time of the lock, i.e.
     * if a transaction acquired locks in the order: S(A), X(B), promote X(A), the
     * lock on A is considered to have been acquired before the lock on B.
     *
     * @throws DuplicateLockRequestException if TRANSACTION already has a
     * NEWLOCKTYPE lock on NAME
     * @throws NoLockHeldException if TRANSACTION has no lock on NAME
     * @throws InvalidLockException if the requested lock type is not a promotion. A promotion
     * from lock type A to lock type B is valid if and only if B is substitutable
     * for A, and B is not equal to A.
     */
    public void promote(BaseTransaction transaction, ResourceName name,
                        LockType newLockType)
            throws DuplicateLockRequestException, NoLockHeldException, InvalidLockException {
        // You may modify any part of this method.

        boolean compatible = true;
        boolean hasTL = false;
        Long transNum = transaction.getTransNum();

        synchronized (this) {
//            throw new UnsupportedOperationException("TODO(hw5_part1): implement");
//            boolean compatible = true;
//            boolean hasTransaction = false;
            for (Map.Entry<Long, List<Lock>> eachTL : this.transactionLocks.entrySet()) {
                Long curtrans = eachTL.getKey();
                List<Lock> thisTransLocks = eachTL.getValue();

                if (curtrans == transNum) {
                    hasTL = true;
                    boolean hasLockOnName = true;
                    for (int i = 0; i < thisTransLocks.size(); i++) {
                        Lock curlock = thisTransLocks.get(i);
                        if (curlock.name.equals(name)) {
                            hasLockOnName = true;
                            if (LockType.substitutable(newLockType, thisTransLocks.get(i).lockType) == false) {
                                throw new InvalidLockException("InvalidLockException");
                            }
                            if (curlock.lockType.equals(newLockType)) {
                                throw new DuplicateLockRequestException("Duplicate lock request exception");
                            }
                        }
                    }

                    if (hasLockOnName == false) {
                        throw new NoLockHeldException("No lock held exception");
                    }

                } else {
                    for (int i = 0; i < thisTransLocks.size(); i++) {
                        Lock curlock = thisTransLocks.get(i);
                        if (curlock.name.equals(name)) {
                            if (LockType.compatible(curlock.lockType, newLockType) == false) {
                                compatible = false;
                                transaction.block();
                                Lock newlock = new Lock(name, newLockType, transNum);
                                LockRequest newRequest = new LockRequest(transaction, newlock);
                                getResourceEntry(name).waitingQueue.addFirst(newRequest);
                            }
                        }
                    }
                }
            }

            if (hasTL == false) {
                throw new NoLockHeldException("No lock held exception");
            }

            if (compatible == true) {
                List<Lock> locklist = this.transactionLocks.get(transNum);
                for (int i = 0; i < locklist.size(); i++) {
                    Lock curlock = locklist.get(i);
                    if (curlock.name.equals(name)) {
//                        LockType oldLockType = list.get(i).lockType;
                        this.transactionLocks.get(transNum).get(i).lockType = newLockType;
                    }
                }
            }
        }
    }

    /**
     * Return the type of lock TRANSACTION has on NAME (return NL if no lock is held).
     */
    public synchronized LockType getLockType(BaseTransaction transaction, ResourceName name) {
//        throw new UnsupportedOperationException("TODO(hw5_part1): implement");
        Long transNum = transaction.getTransNum();
        if (this.transactionLocks.containsKey(transNum)) {
            List<Lock> thelocks = this.transactionLocks.get(transNum);
            for (int i = 0; i < thelocks.size(); i++) {
                Lock curlock = thelocks.get(i);
                if (name == curlock.name) {
                    return curlock.lockType;
                }
            }
        }
        return LockType.NL;
    }

    /**
     * Returns the list of locks held on NAME, in order of acquisition.
     * A promotion or acquire-and-release should count as acquired
     * at the original time.
     */
    public synchronized List<Lock> getLocks(ResourceName name) {
        return new ArrayList<>(resourceEntries.getOrDefault(name, new ResourceEntry()).locks);
    }

    /**
     * Returns the list of locks locks held by
     * TRANSACTION, in order of acquisition. A promotion or
     * acquire-and-release should count as acquired at the original time.
     */
    public synchronized List<Lock> getLocks(BaseTransaction transaction) {
        return new ArrayList<>(transactionLocks.getOrDefault(transaction.getTransNum(),
                Collections.emptyList()));
    }

    /**
     * Create a lock context for the database. See comments at
     * the top of this file and the top of LockContext.java for more information.
     */
    public synchronized LockContext databaseContext() {
        contexts.putIfAbsent("database", new LockContext(this, null, "database"));
        return contexts.get("database");
    }

    /**
     * Create a lock context with no parent. Cannot be called "database".
     */
    public synchronized LockContext orphanContext(Object name) {
        if (name.equals("database")) {
            throw new IllegalArgumentException("cannot create orphan context named 'database'");
        }
        contexts.putIfAbsent(name, new LockContext(this, null, name));
        return contexts.get(name);
    }

}