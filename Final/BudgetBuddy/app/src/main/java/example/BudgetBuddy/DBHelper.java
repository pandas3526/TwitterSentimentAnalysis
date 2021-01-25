package example.BudgetBuddy;
//Authors: Burcu İÇEN-Çağrıhan GÜNAY
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;


class DBHelper extends SQLiteOpenHelper {
    private static final String DATABASE_NAME = "BudgetBuddy.db";

    public static final int ORIGINAL_DATABASE_VERSION = 1;
    public static final int DATABASE_VERSION = 2;


    static class BudgetDbIds {
        public static final String TABLE = "budgets";
        public static final String NAME = "_id";
        public static final String MAX = "max";
    }


    static class TransactionDbIds {
        public static final String TABLE = "transactions";
        public static final String NAME = "_id";
        public static final String TYPE = "type";
        public static final String DESCRIPTION = "description";
        public static final String ACCOUNT = "account";
        public static final String BUDGET = "budget";
        public static final String VALUE = "value";
        public static final String NOTE = "note";
        public static final String DATE = "date";

        public static final int EXPENSE = 1;
        public static final int REVENUE = 2;
    }

    private final Context _context;

    public DBHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
        _context = context;
    }


    private void sendChangeNotification() {
        _context.sendBroadcast(new Intent(TransactionDatabaseChangedReceiver.ACTION_DATABASE_CHANGED));
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(
                "create table  " + BudgetDbIds.TABLE + "(" +
                        BudgetDbIds.NAME + " text primary key," +
                        BudgetDbIds.MAX + " INTEGER not null)");
        // create table for transactions
        db.execSQL("create table " + TransactionDbIds.TABLE + "(" +
                TransactionDbIds.NAME + " INTEGER primary key autoincrement," +
                TransactionDbIds.TYPE + " INTEGER not null," +
                TransactionDbIds.DESCRIPTION + " TEXT not null," +
                TransactionDbIds.ACCOUNT + " TEXT," +
                TransactionDbIds.BUDGET + " TEXT," +
                TransactionDbIds.VALUE + " REAL not null," +
                TransactionDbIds.NOTE + " TEXT," +
                TransactionDbIds.DATE + " INTEGER not null," +
                 " TEXT)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        if (oldVersion < 2 && newVersion >= 2) {
            db.execSQL("ALTER TABLE " + TransactionDbIds.TABLE
                    + " ADD COLUMN "  + " TEXT");
        }
    }


    public boolean insertBudget(final String name, final int max) {
        SQLiteDatabase db = getWritableDatabase();
        boolean result = insertBudget(db, name, max);
        db.close();

        return result;
    }


    public boolean insertBudget(SQLiteDatabase writableDb, final String name, final int max) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(BudgetDbIds.NAME, name);
        contentValues.put(BudgetDbIds.MAX, max);

        final long newId = writableDb.insert(BudgetDbIds.TABLE, null, contentValues);
        return (newId != -1);
    }


    public boolean updateBudget(final String name, final int max) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(BudgetDbIds.MAX, max);

        SQLiteDatabase db = getWritableDatabase();
        int rowsUpdated = db.update(BudgetDbIds.TABLE, contentValues, BudgetDbIds.NAME + "=?",
                new String[]{name});
        db.close();

        return (rowsUpdated == 1);
    }


    public boolean deleteBudget(final String name) {
        SQLiteDatabase db = getWritableDatabase();
        int rowsDeleted = db.delete(BudgetDbIds.TABLE,
                BudgetDbIds.NAME + " = ? ",
                new String[]{name});
        db.close();
        return (rowsDeleted == 1);
    }


    public Budget getBudgetStoredOnly(final String name) {
        SQLiteDatabase db = getReadableDatabase();
        Cursor data = db.rawQuery("select * from " + BudgetDbIds.TABLE +
                " where " + BudgetDbIds.NAME + "=?", new String[]{name});

        Budget budget = null;

        if (data.getCount() == 1) {
            data.moveToFirst();
            String goalName = data.getString(data.getColumnIndexOrThrow(BudgetDbIds.NAME));
            int goalMax = data.getInt(data.getColumnIndexOrThrow(BudgetDbIds.MAX));

            budget = new Budget(goalName, goalMax, 0);
        }

        data.close();
        db.close();

        return budget;
    }


    public List<Budget> getBudgets(long startDateMs, long endDateMs) {
        SQLiteDatabase db = getReadableDatabase();

        final String TOTAL_EXPENSE_COL = "total_expense";
        final String TOTAL_REVENUE_COL = "total_revenue";

        final String BUDGET_ID = BudgetDbIds.TABLE + "." + BudgetDbIds.NAME;
        final String BUDGET_MAX = BudgetDbIds.TABLE + "." + BudgetDbIds.MAX;
        final String TRANS_VALUE = TransactionDbIds.TABLE + "." + TransactionDbIds.VALUE;
        final String TRANS_TYPE = TransactionDbIds.TABLE + "." + TransactionDbIds.TYPE;
        final String TRANS_DATE = TransactionDbIds.TABLE + "." + TransactionDbIds.DATE;
        final String TRANS_BUDGET = TransactionDbIds.TABLE + "." + TransactionDbIds.BUDGET;

        Cursor data = db.rawQuery("select " + BUDGET_ID + ", " + BUDGET_MAX + ", " +
                        "(select total(" + TRANS_VALUE + ") from " + TransactionDbIds.TABLE + " where " +
                        BUDGET_ID + " = " + TRANS_BUDGET + " and " +
                        TRANS_TYPE + " = ? and " +
                        TRANS_DATE + " >= ? and " +
                        TRANS_DATE + " <= ?) " +
                        "as " + TOTAL_EXPENSE_COL + ", " +
                        "(select total(" + TRANS_VALUE + ") from " + TransactionDbIds.TABLE + " where " +
                        BUDGET_ID + " = " + TRANS_BUDGET + " and " +
                        TRANS_TYPE + " = ? and " +
                        TRANS_DATE + " >= ? and " +
                        TRANS_DATE + " <= ?) " +
                        "as " + TOTAL_REVENUE_COL + " " +
                        "from " + BudgetDbIds.TABLE + " order by " + BUDGET_ID,
                new String[]
                        {
                                Integer.toString(TransactionDbIds.EXPENSE),
                                Long.toString(startDateMs),
                                Long.toString(endDateMs),
                                Integer.toString(TransactionDbIds.REVENUE),
                                Long.toString(startDateMs),
                                Long.toString(endDateMs)
                        });

        LinkedList<Budget> budgets = new LinkedList<>();


        Calendar date = Calendar.getInstance();
        date.setTimeInMillis(startDateMs);
        final int MONTHS_PER_YEAR = 12;
        int startMonths = date.get(Calendar.YEAR) * MONTHS_PER_YEAR + date.get(Calendar.MONTH);
        date.setTimeInMillis(endDateMs);
        int endMonths = date.get(Calendar.YEAR) * MONTHS_PER_YEAR + date.get(Calendar.MONTH);
        int totalMonthsInRange = endMonths - startMonths + 1;

        if (data.moveToFirst()) {
            do {
                String name = data.getString(data.getColumnIndexOrThrow(BudgetDbIds.NAME));
                int max = data.getInt(data.getColumnIndexOrThrow(BudgetDbIds.MAX)) * totalMonthsInRange;
                double expenses = data.getDouble(data.getColumnIndexOrThrow(TOTAL_EXPENSE_COL));
                double revenues = data.getDouble(data.getColumnIndexOrThrow(TOTAL_REVENUE_COL));
                double current = expenses - revenues;
                int currentRounded = (int) Math.ceil(current);

                budgets.add(new Budget(name, max, currentRounded));
            } while (data.moveToNext());
        }

        data.close();
        db.close();

        return budgets;
    }


    public Budget getBlankBudget(long startDateMs, long endDateMs) {
        SQLiteDatabase db = getReadableDatabase();

        final String TOTAL_EXPENSE_COL = "total_expense";
        final String TOTAL_REVENUE_COL = "total_revenue";

        final String TRANS_VALUE = TransactionDbIds.TABLE + "." + TransactionDbIds.VALUE;
        final String TRANS_TYPE = TransactionDbIds.TABLE + "." + TransactionDbIds.TYPE;
        final String TRANS_DATE = TransactionDbIds.TABLE + "." + TransactionDbIds.DATE;
        final String TRANS_BUDGET = TransactionDbIds.TABLE + "." + TransactionDbIds.BUDGET;

        Cursor data = db.rawQuery("select " +
                        "(select total(" + TRANS_VALUE + ") from " + TransactionDbIds.TABLE + " where " +
                        TRANS_BUDGET + " = '' and " +
                        TRANS_TYPE + " = ? and " +
                        TRANS_DATE + " >= ? and " +
                        TRANS_DATE + " <= ?) " +
                        "as " + TOTAL_EXPENSE_COL + ", " +
                        "(select total(" + TRANS_VALUE + ") from " + TransactionDbIds.TABLE + " where " +
                        TRANS_BUDGET + " = '' and " +
                        TRANS_TYPE + " = ? and " +
                        TRANS_DATE + " >= ? and " +
                        TRANS_DATE + " <= ?) " +
                        "as " + TOTAL_REVENUE_COL,
                new String[]
                        {
                                Integer.toString(TransactionDbIds.EXPENSE),
                                Long.toString(startDateMs),
                                Long.toString(endDateMs),
                                Integer.toString(TransactionDbIds.REVENUE),
                                Long.toString(startDateMs),
                                Long.toString(endDateMs)
                        });

        int total = 0;

        if (data.moveToFirst()) {
            int expenses = data.getInt(data.getColumnIndexOrThrow(TOTAL_EXPENSE_COL));
            int revenues = data.getInt(data.getColumnIndexOrThrow(TOTAL_REVENUE_COL));
            total = expenses - revenues;
        }

        data.close();
        db.close();

        return new Budget("", 0, total);
    }


    public List<String> getBudgetNames() {
        SQLiteDatabase db = getReadableDatabase();
        Cursor data = db.rawQuery("select " + BudgetDbIds.NAME + " from " + BudgetDbIds.TABLE +
                " ORDER BY " + BudgetDbIds.NAME, null);

        LinkedList<String> budgetNames = new LinkedList<>();

        if (data.moveToFirst()) {
            do {
                String name = data.getString(data.getColumnIndexOrThrow(BudgetDbIds.NAME));

                budgetNames.add(name);
            } while (data.moveToNext());
        }

        data.close();
        db.close();

        return budgetNames;
    }


    public int getBudgetCount() {
        SQLiteDatabase db = getReadableDatabase();
        Cursor data = db.rawQuery("SELECT Count(*) FROM " + BudgetDbIds.TABLE, null);

        int numItems = 0;

        if (data.getCount() == 1) {
            data.moveToFirst();
            numItems = data.getInt(0);
        }

        data.close();
        db.close();

        return numItems;
    }


    public boolean insertTransaction(final int type, final String description, final String account, final String budget,
                                     final double value, final String note, final long dateInMs) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(TransactionDbIds.TYPE, type);
        contentValues.put(TransactionDbIds.DESCRIPTION, description);
        contentValues.put(TransactionDbIds.ACCOUNT, account);
        contentValues.put(TransactionDbIds.BUDGET, budget);
        contentValues.put(TransactionDbIds.VALUE, value);
        contentValues.put(TransactionDbIds.NOTE, note);
        contentValues.put(TransactionDbIds.DATE, dateInMs);

        SQLiteDatabase db = getWritableDatabase();
        long newId = db.insert(TransactionDbIds.TABLE, null, contentValues);
        db.close();

        if (newId != -1) {
            sendChangeNotification();
        }

        return (newId != -1);
    }


    public boolean insertTransaction(SQLiteDatabase writableDb, final int id, final int type, final String description, final String account, final String budget,
                                     final double value, final String note, final long dateInMs) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(TransactionDbIds.NAME, id);
        contentValues.put(TransactionDbIds.TYPE, type);
        contentValues.put(TransactionDbIds.DESCRIPTION, description);
        contentValues.put(TransactionDbIds.ACCOUNT, account);
        contentValues.put(TransactionDbIds.BUDGET, budget);
        contentValues.put(TransactionDbIds.VALUE, value);
        contentValues.put(TransactionDbIds.NOTE, note);
        contentValues.put(TransactionDbIds.DATE, dateInMs);

        long newId = writableDb.insert(TransactionDbIds.TABLE, null, contentValues);

        if (newId != -1) {
            sendChangeNotification();
        }

        return (newId != -1);
    }


    public boolean updateTransaction(final int id, final int type, final String description,
                                     final String account, final String budget, final double value,
                                     final String note, final long dateInMs) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(TransactionDbIds.TYPE, type);
        contentValues.put(TransactionDbIds.DESCRIPTION, description);
        contentValues.put(TransactionDbIds.ACCOUNT, account);
        contentValues.put(TransactionDbIds.BUDGET, budget);
        contentValues.put(TransactionDbIds.VALUE, value);
        contentValues.put(TransactionDbIds.NOTE, note);
        contentValues.put(TransactionDbIds.DATE, dateInMs);

        SQLiteDatabase db = getWritableDatabase();
        int rowsUpdated = db.update(TransactionDbIds.TABLE, contentValues,
                TransactionDbIds.NAME + "=?",
                new String[]{Integer.toString(id)});
        db.close();

        if (rowsUpdated == 1) {
            sendChangeNotification();
        }

        return (rowsUpdated == 1);
    }


    public Transaction getTransaction(final int id) {
        SQLiteDatabase db = getReadableDatabase();
        Cursor data = db.rawQuery("select * from " + TransactionDbIds.TABLE +
                " where " + TransactionDbIds.NAME + "=?", new String[]{Integer.toString(id)});

        Transaction transaction = null;

        if (data.getCount() == 1) {
            data.moveToFirst();
            transaction = Transaction.toTransaction(data);
        }

        data.close();
        db.close();

        return transaction;
    }


    public int getTransactionCount(final int type) {
        SQLiteDatabase db = getReadableDatabase();
        Cursor data = db.rawQuery("SELECT Count(*) FROM " + TransactionDbIds.TABLE +
                " where " + TransactionDbIds.TYPE + "=?", new String[]{Integer.toString(type)});

        int numItems = 0;

        if (data.getCount() == 1) {
            data.moveToFirst();
            numItems = data.getInt(0);
        }

        data.close();
        db.close();

        return numItems;
    }


    public boolean deleteTransaction(final int id) {
        SQLiteDatabase db = getWritableDatabase();
        int rowsDeleted = db.delete(TransactionDbIds.TABLE,
                TransactionDbIds.NAME + " = ? ",
                new String[]{Integer.toString(id)});
        db.close();

        if (rowsDeleted == 1) {
            sendChangeNotification();
        }

        return (rowsDeleted == 1);
    }


    public Cursor getTransactions(int type, String budget, String search, Long startDateMs, Long endDateMs) {
        SQLiteDatabase db = getReadableDatabase();

        LinkedList<String> args = new LinkedList<>();

        String query = "select * from " + TransactionDbIds.TABLE + " where " +
                TransactionDbIds.TYPE + "=" + type;

        if (budget != null) {
            query += " AND " + TransactionDbIds.BUDGET + "=?";
            args.addLast(budget);
        }

        if (search != null) {
            query += " AND ( ";

            String[] items = new String[]{TransactionDbIds.DESCRIPTION, TransactionDbIds.ACCOUNT,
                    TransactionDbIds.VALUE, TransactionDbIds.NOTE};

            for (int index = 0; index < items.length; index++) {
                query += "( " + items[index] + " LIKE ? )";
                if (index < (items.length - 1)) {
                    query += " OR ";
                }
                args.addLast("%" + search + "%");
            }

            query += " )";
        }

        if (startDateMs != null && endDateMs != null) {
            query += " AND " + TransactionDbIds.DATE + " >= ? AND " +
                    TransactionDbIds.DATE + " <= ?";
            args.addLast(Long.toString(startDateMs));
            args.addLast(Long.toString(endDateMs));
        }

        query += " ORDER BY " + TransactionDbIds.DATE + " DESC";

        String[] argArray = args.toArray(new String[args.size()]);

        Cursor res = db.rawQuery(query, argArray);
        return res;
    }


    public Cursor getExpenses() {
        return getTransactions(TransactionDbIds.EXPENSE, null, null, null, null);
    }


    public Cursor getRevenues() {
        return getTransactions(TransactionDbIds.REVENUE, null, null, null, null);
    }



}
