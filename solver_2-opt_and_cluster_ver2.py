#---- わかりやすく書き直したコード ----#

#!/usr/bin/env python3
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import print_tour, read_input


class TSPSolver:
    def __init__(self, cities, split_num):
        self.cities = cities
        self.route_graph = [[] for _ in range(len(self.cities))]  # 道順を表すグラフ. ノードi に隣接するノードj を route_graph[i] に格納する
        self.clusters = []     # 各クラスターに含まれる id の配列. 
        self.split_num = split_num

    # ------------------------------------------------------------------
    # 1. クラスタリング
    # ------------------------------------------------------------------
    def is_city_in_range(self, city_id, cluster_range):
        x, y = self.cities[city_id]
        return (
            cluster_range['x_min'] <= x < cluster_range['x_max']
            and cluster_range['y_min'] <= y < cluster_range['y_max']
        )

    def make_cluster_ranges(self):
        total_x_min = min(x for x, _ in self.cities) - 1
        total_x_max = max(x for x, _ in self.cities) + 1
        total_y_min = min(y for _, y in self.cities) - 1
        total_y_max = max(y for _, y in self.cities) + 1

        x_step = (total_x_max - total_x_min) / self.split_num
        y_step = (total_y_max - total_y_min) / self.split_num

        cluster_ranges = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                cluster_ranges.append({
                    'x_min': total_x_min + x_step * i,
                    'x_max': total_x_min + x_step * (i + 1),
                    'y_min': total_y_min + y_step * j,
                    'y_max': total_y_min + y_step * (j + 1),
                })
        return cluster_ranges

    def assign_cities_to_clusters(self, cluster_ranges):
        raw_clusters = [[] for _ in range(len(cluster_ranges))]
        for city_id in range(len(self.cities)):
            for cluster_id, cluster_range in enumerate(cluster_ranges):
                if self.is_city_in_range(city_id, cluster_range):
                    raw_clusters[cluster_id].append(city_id)
                    break
        return raw_clusters

    def merge_small_clusters(self, raw_clusters):
        merged = []
        for cluster in raw_clusters:
            if len(cluster) < 4:
                if merged:
                    merged[-1].extend(cluster)
                else:
                    merged.append(cluster)
            else:
                merged.append(cluster)
        return merged

    def make_clusters(self):
        cluster_ranges = self.make_cluster_ranges()
        raw_clusters = self.assign_cities_to_clusters(cluster_ranges)
        self.clusters = self.merge_small_clusters(raw_clusters)

    # ------------------------------------------------------------------
    # 距離・グラフの基本操作
    # ------------------------------------------------------------------
    def calc_dist_of_cities(self, city_id_1, city_id_2):
        x1, y1 = self.cities[city_id_1]
        x2, y2 = self.cities[city_id_2]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def connect_cities(self, city_id_1, city_id_2):
        self.route_graph[city_id_1].append(city_id_2)
        self.route_graph[city_id_2].append(city_id_1)

    def next_path(self, path):
        prev_id, cur_id = path
        if self.route_graph[cur_id][0] != prev_id:
            next_id = self.route_graph[cur_id][0]
        else:
            next_id = self.route_graph[cur_id][1]
        return (cur_id, next_id)

    ## id から cur_neighbor までの有向辺を削除し new_neighbor までの有向辺を追加する
    def change_path(self, city_id, cur_neighbor, new_neighbor):
        if self.route_graph[city_id][0] == cur_neighbor:
            self.route_graph[city_id][0] = new_neighbor
        else:
            self.route_graph[city_id][1] = new_neighbor

    # ------------------------------------------------------------------
    # 2. 各クラスタ内の局所巡回路作成
    # ------------------------------------------------------------------
    def greedy(self, cluster_id):
        INF = 10**9
        visited = [False] * len(self.cities)

        cluster = self.clusters[cluster_id]
        root_id = cluster[0]
        cur_city_id = root_id
        visited[root_id] = True

        while True:
            min_dist = INF
            next_city_id = -1

            for city_id in cluster:
                if visited[city_id]:
                    continue
                dist = self.calc_dist_of_cities(cur_city_id, city_id)
                if dist < min_dist:
                    min_dist = dist
                    next_city_id = city_id

            if next_city_id == -1:
                break

            self.connect_cities(cur_city_id, next_city_id)
            cur_city_id = next_city_id
            visited[cur_city_id] = True

        # 最後に root_id にもどるルートを追加
        self.connect_cities(cur_city_id, root_id)

    ## path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 と交換した方が巡回路長が短くなる場合交換する. 
    ## 返り値は, 処理後のパスと, 交換したかを表す bool 値. 
    ## id1⇔id4, id2⇔id3 と交換するとグラフが連結でなくなるので不可.
    def swap_path_if_shorter(self, path1, path2):
        id1, id2 = path1
        id3, id4 = path2

        cur_path_length = self.calc_dist_of_cities(id1, id2) + self.calc_dist_of_cities(id3, id4)
        new_path_length = self.calc_dist_of_cities(id1, id3) + self.calc_dist_of_cities(id2, id4)

        if cur_path_length > new_path_length:
            self.change_path(id1, id2, id3)
            self.change_path(id2, id1, id4)
            self.change_path(id3, id4, id1)
            self.change_path(id4, id3, id2)
            return ((id1, id3), (id2, id4), True)
        return (path1, path2, False)

    def apply_2opt_once(self, cluster_id):
        if len(self.clusters[cluster_id]) < 4:
            return False

        is_graph_changed = False
        root_id = self.clusters[cluster_id][0]
        path1 = (root_id, self.route_graph[root_id][0])

        while True:
            path2 = self.next_path(path1)
            path2 = self.next_path(path2)
            if path2[0] == root_id:
                break

            while path2[0] != root_id:
                path1, path2, is_path_changed = self.swap_path_if_shorter(path1, path2)
                is_graph_changed = is_graph_changed or is_path_changed
                path2 = self.next_path(path2)
            path1 = self.next_path(path1)

        return is_graph_changed

    def apply_2opt(self, cluster_id):
        for _ in range(5):
            changed_graph = self.apply_2opt_once(cluster_id)
            if not changed_graph:
                break

    def solve_cluster(self, cluster_id):
        self.greedy(cluster_id)
        self.apply_2opt(cluster_id)

    # ------------------------------------------------------------------
    # 巡回路の変換・表示
    # ------------------------------------------------------------------
    ## root_id からグラフに沿って id をめぐって, 通った id を配列に書き出す. グラフが連結でない場合は, root_id を含む連結部分のみ書き出される.
    def graph_to_city_id_tour(self, root_id=0):
        city_id_tour = []
        prev_id = root_id
        city_id_tour.append(prev_id)
        cur_id = self.route_graph[prev_id][0]

        while cur_id != root_id:
            city_id_tour.append(cur_id)
            prev_id, cur_id = self.next_path((prev_id, cur_id))

        return city_id_tour

    ## 指定のクラスターを巡回するパスを配列に書き出す. 
    def graph_to_path_tour(self, cluster_id):
        path_tour = []
        root_id = self.clusters[cluster_id][0]
        cur_path = (root_id, self.route_graph[root_id][0])
        while True:
            path_tour.append(cur_path)
            cur_path = self.next_path(cur_path)
            if cur_path[0] == root_id:
                break
        return path_tour

    # ------------------------------------------------------------------
    # 3. クラスタ同士の結合
    # ------------------------------------------------------------------
    ## path1 と path2（2つは別のクラスターのパス）を繋ぎ替えた場合に, 全体の巡回路の長さがどれくらい増えるか計算する. 増えるとき正の数を返す.
    ## また, path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 と繋ぎ替えても id1⇔id4, id2⇔id3 と繋ぎ替えてもよいので,
    ## id1⇔id3, id2⇔id4 とした方が全体の巡回路が短くなる場合 "1to3_and_2to4" を, 他方の場合は "1to4_and_2to3" を返す.
    def calc_decrease_dist_and_better_swapping_way(self, path1, path2):
        id1, id2 = path1
        id3, id4 = path2

        cur_dist = self.calc_dist_of_cities(id1, id2) + self.calc_dist_of_cities(id3, id4)
        dist_1to3_2to4 = self.calc_dist_of_cities(id1, id3) + self.calc_dist_of_cities(id2, id4)
        dist_1to4_2to3 = self.calc_dist_of_cities(id1, id4) + self.calc_dist_of_cities(id2, id3)

        if dist_1to3_2to4 < dist_1to4_2to3:
            return (dist_1to3_2to4 - cur_dist, "1to3_and_2to4")
        return (dist_1to4_2to3 - cur_dist, "1to4_and_2to3")

    def join_clusters(self):
        INF = 10**9
        base_cluster_id = 0
        disconnected_cluster_ids = list(range(1, len(self.clusters)))

        while disconnected_cluster_ids:
            best_choice = {
                'decrease_dist': INF,
            }

            for path0 in self.graph_to_path_tour(base_cluster_id):
                for cluster_id in disconnected_cluster_ids:
                    for path1 in self.graph_to_path_tour(cluster_id):
                        decrease_dist, way = self.calc_decrease_dist_and_better_swapping_way(path0, path1)
                        if decrease_dist < best_choice['decrease_dist']:
                            best_choice = {
                                'decrease_dist': decrease_dist,
                                'path0': path0,
                                'path1': path1,
                                'cluster_id': cluster_id,
                                'way': way,
                            }

            if best_choice['way'] == "1to3_and_2to4":
                id1, id2 = best_choice['path0']
                id3, id4 = best_choice['path1']
                self.change_path(id1, id2, id3)
                self.change_path(id2, id1, id4)
                self.change_path(id3, id4, id1)
                self.change_path(id4, id3, id2)
            elif best_choice['way'] == "1to4_and_2to3":
                id1, id2 = best_choice['path0']
                id3, id4 = best_choice['path1']
                self.change_path(id1, id2, id4)
                self.change_path(id2, id1, id3)
                self.change_path(id3, id4, id2)
                self.change_path(id4, id3, id1)
            else:
                return

            disconnected_cluster_ids.remove(best_choice['cluster_id'])

        self.clusters = [list(range(len(self.cities)))]

    # ------------------------------------------------------------------
    # 補助表示・評価
    # ------------------------------------------------------------------
    def plot_graph(self, num=""):
        for cluster in self.clusters:
            root_id = cluster[0]
            city_id_tour = self.graph_to_city_id_tour(root_id)

            x = [self.cities[i][0] for i in city_id_tour]
            y = [-self.cities[i][1] for i in city_id_tour]
            x.append(self.cities[root_id][0])
            y.append(-self.cities[root_id][1])

            plt.plot(x, y, marker='.', markersize=5)
        plt.savefig(f'graph_{num}.png')
        plt.close()

    def calc_tour_length(self):
        root_id = 0
        path_tour = self.graph_to_path_tour(root_id)
        tour_length = 0
        for path in path_tour:
            tour_length += self.calc_dist_of_cities(path[0], path[1])
        return tour_length

    # ------------------------------------------------------------------
    # 5. 全体の流れ
    # ------------------------------------------------------------------
    def solve(self):
        self.make_clusters()

        for cluster_id in range(len(self.clusters)):
            self.solve_cluster(cluster_id)
        # self.plot_graph("cluster")

        self.join_clusters()
        self.apply_2opt(0)
        # self.plot_graph("joint")

        return (self.graph_to_city_id_tour(), self.calc_tour_length())

def solve(cities):
    INF = 10**9
    best_answer = ([], INF)
    for i in range(1,10):
        tsp_solver = TSPSolver(cities, i)
        answer = tsp_solver.solve()
        if answer[1] < best_answer[1]:
            best_answer = answer
    return best_answer[0]



if __name__ == '__main__':
    assert len(sys.argv) > 1
    tour = solve(read_input(sys.argv[1]))
    print_tour(tour)
