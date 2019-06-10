package edu.berkeley.cs186.database.concurrency;

public enum LockType {
    S,   // shared
    X,   // exclusive
    IS,  // intention shared
    IX,  // intention exclusive
    SIX, // shared intention exclusive
    NL;  // no lock held

    /**
     * This method checks whether lock types A and B are compatible with
     * each other. If a transaction can hold lock type A on a resource
     * at the same time another transaction holds lock type B on the same
     * resource, the lock types are compatible.
     */
    public static boolean compatible(LockType a, LockType b) {
        if (a == null || b == null) {
            throw new NullPointerException("null lock type");
        }
//        throw new UnsupportedOperationException("TODO(hw5_part1): implement");
        boolean compatible = true;

        if (a == S) {
            if (b == X || b == IX || b == SIX) {
                compatible = false;
            }

        } else if (a == X) {
            if (b != null) {
                compatible = false;
            }

        } else if (a == IS) {
            if (b == X) {
                compatible = false;
            }

        } else if (a == IX) {
            if (b == S || b == X || b == SIX) {
                compatible = false;
            }

        } else if (a == SIX) {
            if (b == S || b == X || b == IX || b == SIX) {
                compatible = false;
            }

        } else if (a == NL) {
            compatible = true;
        }

        if (b == NL) {
            compatible = true;
        }


        return compatible;
    }

    /**
     * This method returns the least permissive lock on the parent resource
     * that must be held for a lock of type A to be granted.
     */
    public static LockType parentLock(LockType a) {
        if (a == null) {
            throw new NullPointerException("null lock type");
        }
//        throw new UnsupportedOperationException("TODO(hw5_part1): implement");
        LockType locktype = null;

        if (a == S) {
            locktype = IS;

        } else if (a == X) {
            locktype = IX;

        } else if (a == IS) {
            locktype = IX;

        } else if (a == IX) {
            locktype = IX;

        } else if (a == SIX) {
            locktype = IX;

        } else if (a == NL) {
            locktype = NL;

        }

        return locktype;
    }

    /**
     * This method returns whether a lock can be used for a situation
     * requiring another lock (e.g. an S lock can be substituted with
     * an X lock, because an X lock allows the transaction to do everything
     * the S lock allowed it to do).
     */
    public static boolean substitutable(LockType substitute, LockType required) {
        if (required == null || substitute == null) {
            throw new NullPointerException("null lock type");
        }
//        throw new UnsupportedOperationException("TODO(hw5_part1): implement");
        boolean substitutable = false;

        if (required == S) {
            if (substitute == S || substitute == X || substitute == SIX) {
                substitutable = true;

            }

        } else if (required == X) {
            if (substitute == X) {
                substitutable = true;
            }

        } else if (required == IS) {
            if (substitute == S || substitute == X || substitute == IS || substitute == IX || substitute == SIX) {
                substitutable = true;
            }

        } else if (required == IX) {
            if (substitute == X || substitute == IX || substitute == SIX) {
                substitutable = true;
            }

        } else if (required == SIX) {
            if (substitute == X || substitute == SIX) {
                substitutable = true;
            }

        }

        return substitutable;


    }

    @Override
    public String toString() {
        switch (this) {
        case S: return "S";
        case X: return "X";
        case IS: return "IS";
        case IX: return "IX";
        case SIX: return "SIX";
        case NL: return "NL";
        default: throw new UnsupportedOperationException("bad lock type");
        }
    }
};

