package example.BudgetBuddy;
//Author: Çağrıhan GÜNAY

import android.app.SearchManager;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.support.design.widget.TabLayout;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.ViewPager;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.Menu;
import android.view.MenuItem;

public class TransactionActivity extends AppCompatActivity {
    private TransactionDatabaseChangedReceiver _dbChanged;

    private boolean _currentlySearching = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.transaction_activity);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
        }

        _dbChanged = new TransactionDatabaseChangedReceiver();
        this.registerReceiver(_dbChanged, new IntentFilter(TransactionDatabaseChangedReceiver.ACTION_DATABASE_CHANGED));

        String search = getIntent().getStringExtra(SearchManager.QUERY);
        resetView(search);
    }

    private void resetView(String search) {
        TabLayout tabLayout = (TabLayout) findViewById(R.id.tabLayout);
        tabLayout.removeAllTabs();
        tabLayout.addTab(tabLayout.newTab().setText(R.string.expensesTitle));
        tabLayout.addTab(tabLayout.newTab().setText(R.string.revenuesTitle));
        tabLayout.setTabGravity(TabLayout.GRAVITY_FILL);

        final ViewPager viewPager = (ViewPager) findViewById(R.id.pager);
        final PagerAdapter adapter = new TransactionPagerAdapter
                (getSupportFragmentManager(), tabLayout.getTabCount());
        viewPager.setAdapter(adapter);

        viewPager.addOnPageChangeListener(new TabLayout.TabLayoutOnPageChangeListener(tabLayout));
        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewPager.setCurrentItem(tab.getPosition());
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {

            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) {

            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();

        if (_dbChanged.hasChanged() || Intent.ACTION_SEARCH.equals(getIntent().getAction())) {
            String search = null;




            resetView(search);
            _dbChanged.reset();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.transaction_menu, menu);

        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_add) {
            Intent i = new Intent(getApplicationContext(), TransactionViewActivity.class);
            final Bundle b = new Bundle();
            b.putInt("type", getCurrentTabType());
            i.putExtras(b);
            startActivity(i);
            return true;
        }


        if (id == android.R.id.home) {
            finish();
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private int getCurrentTabType() {
        TabLayout tabLayout = (TabLayout) findViewById(R.id.tabLayout);

        if (tabLayout.getSelectedTabPosition() == 0) {
            return DBHelper.TransactionDbIds.EXPENSE;
        } else {
            return DBHelper.TransactionDbIds.REVENUE;
        }
    }

    @Override
    public void onDestroy() {
        this.unregisterReceiver(_dbChanged);
        super.onDestroy();
    }
}