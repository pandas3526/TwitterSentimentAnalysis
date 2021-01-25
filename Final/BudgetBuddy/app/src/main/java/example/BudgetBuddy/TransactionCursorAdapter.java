package example.BudgetBuddy;
//Author: Burcu İÇEN

import android.content.Context;
import android.database.Cursor;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CursorAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Locale;

class TransactionCursorAdapter extends CursorAdapter {
    public TransactionCursorAdapter(Context context, Cursor cursor) {
        super(context, cursor, 0);
    }

    private final DateFormat DATE_FORMATTER = SimpleDateFormat.getDateInstance();

    static class ViewHolder {
        TextView nameField;
        TextView valueField;
        TextView dateField;
        TextView budgetField;
        TextView note;
        View noteLayout;
    }


    @Override
    public View newView(Context context, Cursor cursor, ViewGroup parent) {
        View view = LayoutInflater.from(context).inflate(R.layout.transaction_layout, parent, false);

        ViewHolder holder = new ViewHolder();
        holder.nameField = (TextView) view.findViewById(R.id.name);
        holder.valueField = (TextView) view.findViewById(R.id.value);
        holder.dateField = (TextView) view.findViewById(R.id.date);
        holder.budgetField = (TextView) view.findViewById(R.id.budget);
        holder.note = (TextView) view.findViewById(R.id.note);
        holder.noteLayout = view.findViewById(R.id.noteLayout);
        view.setTag(holder);

        return view;
    }


    @Override
    public void bindView(View view, Context context, Cursor cursor) {
        ViewHolder holder = (ViewHolder) view.getTag();

        Transaction transaction = Transaction.toTransaction(cursor);
        holder.nameField.setText(transaction.description);
        holder.valueField.setText(String.format(Locale.US, "%.2f", transaction.value));
        holder.budgetField.setText(transaction.budget);
        holder.dateField.setText(DATE_FORMATTER.format(transaction.dateMs));



        if (transaction.note.isEmpty()) {
            holder.noteLayout.setVisibility(View.GONE);
            holder.note.setText("");
        } else {
            holder.noteLayout.setVisibility(View.VISIBLE);
            holder.note.setText(transaction.note);
        }
    }
}
