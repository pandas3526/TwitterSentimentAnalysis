package example.BudgetBuddy;
//Author: Çağrıhan GÜNAY

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;


public class TransactionDatabaseChangedReceiver extends BroadcastReceiver {
    public static final String ACTION_DATABASE_CHANGED = "example.BudgetBuddy.TRANSACTION_DATABASE_CHANGED";

    private boolean _hasChanged = false;

    @Override
    public void onReceive(Context context, Intent intent) {
        _hasChanged = true;
    }

    public boolean hasChanged() {
        return _hasChanged;
    }

    public void reset() {
        _hasChanged = false;
    }
}