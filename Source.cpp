#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	//ifstream cin(".in");
	//ofstream cout(".out");
	int N; cin >> N;
	int act[501];
	for (int i = 0; i < N; i++) {
		cin >> act[501];
	}
	int pre[501];
	for (int i = 0; i < N; i++) {
		set<int> sums;
		pre[0] = act[0];
		for (int j = 1; j < i; j++) {
			pre[j] = act[j] + pre[j-1];
		}
		for (int j = i + 1; j < N; j++) {
			
		}
	}
}