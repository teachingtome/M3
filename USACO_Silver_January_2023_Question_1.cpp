#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long
bool done(vector<bool>& visited, vector<int>& parent) {
	for (int i = 'A'; i <= 'z'; i++) {
		if (i >= 91 && i <= 96) {
			continue;
		}
		if ((!visited[i] && parent[i] != i) && parent[i] != 0)  {
			return false;
		}
	}
	return true;
}
int ff(int on, vector<bool>& visited, vector<int>& parent, vector<int>& numFeed, bool usedAll, bool fullCycle, int tot) {
	if (numFeed[on] > 1) {
		fullCycle = false;
	}
	if (visited[on] || parent[on] == 0 || parent[on] == on) {
		if (!fullCycle || parent[on] == 0 || parent[on] == on) {
			return tot;
		}
		if (!usedAll) {
			return tot + 1;
		}
		return -1;
	}
	tot++;
	visited[on] = true;
	ff(parent[on], visited, parent, numFeed, usedAll, fullCycle, tot);
}
void solve() {
	vector<int> parent(130, 0);
	vector<int> numFeed(130, 0);
	vector<bool> used(130, false);
	string s; cin >> s; string s1; cin >> s1;
	for (int i = 0; i < s.length(); i++) {
		used[s1[i]] = true;
		if (parent[s[i]] == 0) {
			parent[s[i]] = s1[i];
			numFeed[s1[i]]++;
		}
		else if(parent[s[i]] != s1[i]){
			cout << -1 << "\n";
			return;
		}
	}
	bool usedAll = true;
	for (int i = 'A'; i <= 'z'; i++) {
		if (i >= 91 && i <= 96) {
			continue;
		}
		if (used[i] == 0) {
			usedAll = false;
			break;
		}
	}
	vector<bool> visited(130, false);
	int tot = 0;
	while (!done(visited, parent)) {
		int minFlow = 10000000;
		int minFlowChar = 0;
		for (int i = 'A'; i <= 'z'; i++) {
			if (i >= 91 && i <= 96) {
				continue;
			}
			if (!visited[i] && parent[i] != i && parent[i] != 0) {
				if (numFeed[i] < minFlow) {
					minFlow = numFeed[i];
					minFlowChar = i;
				}
			}
		}
		int val = ff(minFlowChar, visited, parent, numFeed, usedAll, true, 0);
		if (val == -1) {
			cout << -1 << "\n";
			return;
		}
		tot += val;
	}
	cout << tot << "\n";
}
int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	//ifstream cin(".in");
	//ofstream cout(".out");
	int T; cin >> T;
	while (T--) {
		solve();
	}
}
