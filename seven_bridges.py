from manim import *

from typing import Hashable


#####
# from Nipun Ramk's https://github.com/nipunramk/Reducible

import itertools as it

REDUCIBLE_PURPLE_DARK_FILL = "#331B5D"
REDUCIBLE_PURPLE_DARKER = "#3B0893"
REDUCIBLE_PURPLE = "#8c4dfb"
REDUCIBLE_VIOLET = "#d7b5fe"

REDUCIBLE_BLUE = "#650FFA"
REDUCIBLE_BLUE_DARKER = "#02034E"

REDUCIBLE_YELLOW = "#ffff5c"
REDUCIBLE_YELLOW_DARKER = "#7F7F2D"

REDUCIBLE_GREEN_LIGHTER = "#00cc70"
REDUCIBLE_GREEN = "#008f4f"
REDUCIBLE_GREEN_DARKER = "#004F2C"

REDUCIBLE_WARM_BLUE = "#08B6CE"
REDUCIBLE_WARM_BLUE_DARKER = "#044263"

REDUCIBLE_ORANGE = "#FFB413"
REDUCIBLE_ORANGE_DARKER = "#714400"

REDUCIBLE_CHARM = "#FF5752"
REDUCIBLE_CHARM_DARKER = "#6F001F"

REDUCIBLE_FONT = "CMU Serif"
REDUCIBLE_MONO = "SF Mono"


class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class MarkovChain:
    def __init__(
        self,
        states: int,
        edges: list[tuple[int, int]],
        transition_matrix=None,
        dist=None,
    ):
        """
        @param: states -- number of states in Markov Chain
        @param: edges -- list of tuples (u, v) for a directed edge u to v, u in range(0, states), v in range(0, states)
        @param: transition_matrix -- custom np.ndarray matrix of transition probabilities for all states in Markov chain
        @param: dist -- initial distribution across states, assumed to be uniform if none
        """
        self.states = range(states)
        self.edges = edges
        self.adj_list = {}
        for state in self.states:
            self.adj_list[state] = []
            for u, v in edges:
                if u == state:
                    self.adj_list[state].append(v)

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Assume default transition matrix is uniform across all outgoing edges
            self.transition_matrix = np.zeros((states, states))
            for state in self.states:
                neighbors = self.adj_list[state]
                for neighbor in neighbors:
                    self.transition_matrix[state][neighbor] = 1 / len(neighbors)

        # handle sink nodes to point to itself
        for i, row in enumerate(self.transition_matrix):
            if np.sum(row) == 0:
                self.transition_matrix[i][i] = 1

        if dist is not None:
            self.dist = dist
        else:
            self.dist = np.array(
                [1 / len(self.states) for _ in range(len(self.states))]
            )

        self.starting_dist = self.dist

    def get_states(self):
        return list(self.states)

    def get_edges(self):
        return self.edges

    def get_adjacency_list(self):
        return self.adj_list

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_current_dist(self):
        return self.dist

    def update_dist(self):
        """
        Performs one step of the markov chain
        """
        self.dist = np.dot(self.dist, self.transition_matrix)

    def get_true_stationary_dist(self):
        dist = np.linalg.eig(np.transpose(self.transition_matrix))[1][:, 0]
        return dist / sum(dist)

    def set_starting_dist(self, starting_dist):
        self.starting_dist = starting_dist
        self.dist = starting_dist

    def get_starting_dist(self):
        return self.starting_dist

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix


class CustomCurvedArrow(CurvedArrow):
    def __init__(self, start, end, tip_length=0.15, **kwargs):
        super().__init__(start, end, **kwargs)
        self.pop_tips()
        self.add_tip(
            tip_shape=ArrowTriangleFilledTip,
            tip_length=tip_length,
            at_start=False,
        )
        self.tip.z_index = -100

    def set_opacity(self, opacity, family=True):
        return super().set_opacity(opacity, family)

    @override_animate(set_opacity)
    def _set_opacity_animation(self, opacity=1, anim_args=None):
        if anim_args is None:
            anim_args = {}

        animate_stroke = self.animate.set_stroke(opacity=opacity)
        animate_tip = self.tip.animate.set_opacity(opacity)

        return AnimationGroup(*[animate_stroke, animate_tip])


class MarkovChainGraph(Graph):
    def __init__(
        self,
        markov_chain: MarkovChain,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE,
            "fill_opacity": 0.5,
        },
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        enable_curved_double_arrows=True,
        labels=True,
        state_color_map=None,
        **kwargs,
    ):
        self.markov_chain = markov_chain
        self.enable_curved_double_arrows = enable_curved_double_arrows

        self.default_curved_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
            "radius": 4,
        }

        self.default_straight_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "max_tip_length_to_length_ratio": 0.06,
            "stroke_width": 3,
        }
        self.state_color_map = state_color_map

        if labels:
            labels = {
                k: CustomLabel(str(k), scale=0.6) for k in markov_chain.get_states()
            }

        if self.state_color_map:
            new_vertex_config = {}
            for state in markov_chain.get_states():
                new_vertex_config[state] = vertex_config.copy()
                new_vertex_config[state]["stroke_color"] = self.state_color_map[state]
                new_vertex_config[state]["fill_color"] = self.state_color_map[state]

            vertex_config = new_vertex_config

        self.labels = {}

        super().__init__(
            markov_chain.get_states(),
            markov_chain.get_edges(),
            vertex_config=vertex_config,
            labels=labels,
            **kwargs,
        )

        self._graph = self._graph.to_directed()
        self.remove_edges(*self.edges)

        self.add_markov_chain_edges(
            *markov_chain.get_edges(),
            straight_edge_config=straight_edge_config,
            curved_edge_config=curved_edge_config,
        )

        self.clear_updaters()

        # this updater makes sure the edges remain connected
        # even when states move around
        def update_edges(graph):
            for (u, v), edge in graph.edges.items():
                v_c = self.vertices[v].get_center()
                u_c = self.vertices[u].get_center()
                vec = v_c - u_c
                unit_vec = vec / np.linalg.norm(vec)

                u_radius = self.vertices[u].width / 2
                v_radius = self.vertices[v].width / 2

                arrow_start = u_c + unit_vec * u_radius
                arrow_end = v_c - unit_vec * v_radius
                edge.put_start_and_end_on(arrow_start, arrow_end)

        self.add_updater(update_edges)
        update_edges(self)

    def add_edge_buff(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[Mobject] = None,
        edge_config: dict = None,
    ):
        """
        Custom function to add edges to our Markov Chain,
        making sure the arrowheads land properly on the states.
        """
        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        v_c = self.vertices[v].get_center()
        u_c = self.vertices[u].get_center()
        vec = v_c - u_c
        unit_vec = vec / np.linalg.norm(vec)

        if self.enable_curved_double_arrows:
            arrow_start = u_c + unit_vec * self.vertices[u].radius
            arrow_end = v_c - unit_vec * self.vertices[v].radius
        else:
            arrow_start = u_c
            arrow_end = v_c
            edge_config["buff"] = self.vertices[u].radius

        edge_mobject = edge_type(
            start=arrow_start, end=arrow_end, z_index=-100, **edge_config
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def add_markov_chain_edges(
        self,
        *edges: tuple[Hashable, Hashable],
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        **kwargs,
    ):
        """
        Custom function for our specific case of Markov Chains.
        This function aims to make double arrows curved when two nodes
        point to each other, leaving the other ones straight.
        Parameters
        ----------
        - edges: a list of tuples connecting states of the Markov Chain
        - curved_edge_config: a dictionary specifying the configuration
        for CurvedArrows, if any
        - straight_edge_config: a dictionary specifying the configuration
        for Arrows
        """

        if curved_edge_config is not None:
            curved_config_copy = self.default_curved_edge_config.copy()
            curved_config_copy.update(curved_edge_config)
            curved_edge_config = curved_config_copy
        else:
            curved_edge_config = self.default_curved_edge_config.copy()

        if straight_edge_config is not None:
            straight_config_copy = self.default_straight_edge_config.copy()
            straight_config_copy.update(straight_edge_config)
            straight_edge_config = straight_config_copy
        else:
            straight_edge_config = self.default_straight_edge_config.copy()

        print(straight_edge_config)

        edge_vertices = set(it.chain(*edges))
        new_vertices = [v for v in edge_vertices if v not in self.vertices]
        added_vertices = self.add_vertices(*new_vertices, **kwargs)

        edge_types_dict = {}
        for e in edges:
            if self.enable_curved_double_arrows and (e[1], e[0]) in edges:
                edge_types_dict.update({e: (CustomCurvedArrow, curved_edge_config)})

            else:
                edge_types_dict.update({e: (Arrow, straight_edge_config)})

        added_mobjects = sum(
            (
                self.add_edge_buff(
                    edge,
                    edge_type=e_type_and_config[0],
                    edge_config=e_type_and_config[1],
                ).submobjects
                for edge, e_type_and_config in edge_types_dict.items()
            ),
            added_vertices,
        )

        return self.get_group_class()(*added_mobjects)

    def get_transition_labels(self, scale=0.3, round_val=True):
        """
        This function returns a VGroup with the probability that each
        each state has to transition to another state, based on the
        Chain's transition matrix.
        It essentially takes each edge's probability and creates a label to put
        on top of it, for easier indication and explanation.
        This function returns the labels already set up in a VGroup, ready to just
        be created.
        """
        tm = self.markov_chain.get_transition_matrix()

        labels = VGroup()
        for s in range(len(tm)):
            for e in range(len(tm[0])):
                if s != e and tm[s, e] != 0:
                    edge_tuple = (s, e)
                    matrix_prob = tm[s, e]

                    if round_val and round(matrix_prob, 2) != matrix_prob:
                        matrix_prob = round(matrix_prob, 2)

                    label = (
                        Text(str(matrix_prob), font=REDUCIBLE_MONO)
                        .set_stroke(BLACK, width=8, background=True, opacity=0.8)
                        .scale(scale)
                        .move_to(self.edges[edge_tuple].point_from_proportion(0.2))
                    )

                    labels.add(label)
                    self.labels[edge_tuple] = label

        def update_labels(graph):
            for e, l in graph.labels.items():
                l.move_to(graph.edges[e].point_from_proportion(0.2))

        self.add_updater(update_labels)

        return labels


class SevenBridges(Scene):
    """
    This class represents the Seven Bridges of Königsberg problem.
    It creates a scene with a city outline, rivers, and bridges.
    It also includes a method to animate a dot traversing the bridges.
    ref: https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg
    """

    def construct(self):
        # Create a rectangle to represent the city outline
        self.city_outline = Rectangle(width=11, height=6, color=WHITE)
        # Create a title for the scene
        self.title = Tex("Seven Bridges of Königsberg")
        self.title.scale(1.2)
        self.title.next_to(self.city_outline, UP)

        # Create the rivers
        # Main river path
        self.main_river_path = ParametricFunction(
            lambda t: np.array(
                [t - 1, -0.35 * (t - 1) ** 2 + 0.7, 0]
            ),  # Adjust coefficients to change the arc
            t_range=[-3.6, 5.6],
            stroke_width=40,
            color=BLUE,
        )
        self.main_river_path.shift(LEFT)

        # Transverse river path
        self.transverse_river_path = Line(
            start=np.array([-2.1, -0.7, 0]),
            end=np.array([2.1, -0.7, 0]),
            stroke_width=40,
            color=BLUE,
        )
        self.transverse_river_path.shift(LEFT)

        # Fork river path
        self.fork_river_path = Line(
            start=np.array([0, 0.7, 0]),
            end=np.array([9.9, 0.7, 0]),
            stroke_width=40,
            color=BLUE,
        )
        self.fork_river_path.shift(LEFT)

        # Group all rivers together
        self.reviers_group = VGroup(
            self.main_river_path, self.transverse_river_path, self.fork_river_path
        )

        # Create the bridges
        bridge_color = "#F8DE4E"  # Color of the bridges
        bridge_width = 30  # Width of the bridge
        bridge_length = 0.45  # Length of the bridge, adjust as needed

        # Bridge between the upper land and the right land
        bridge1 = Line(
            start=np.array([2, 0.5 - 0.1, 0]),
            end=np.array([2, 0.5 + bridge_length, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge1",
        )

        # Bridge between the upper land and the central island
        bridge2 = Line(
            start=np.array([-1.9, 0.1, 0.1]),
            end=np.array([-2.2, 0.1 + bridge_length, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge2",
        )
        bridge3 = Line(
            start=np.array([-1, 0.5, 0]),
            end=np.array([-1.3, 0.5 + bridge_length + 0.1, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge3",
        )

        # Bridge between the lower land and the right land
        bridge4 = Line(
            start=np.array([0.9, -1.1, 0]),
            end=np.array([1.3, -1.1 + bridge_length, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge4",
        )

        # Bridge between the lower land and the central island
        bridge5 = Line(
            start=np.array([0.1, -0.5 - bridge_length, 0]),
            end=np.array([0.1, -0.5 + 0.1, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge5",
        )
        bridge6 = Line(
            start=np.array([-1.7, -0.5 - bridge_length, 0]),
            end=np.array([-1.7, -0.5 + 0.1, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge6",
        )

        # Bridge between the right land and the central island
        bridge7 = Line(
            start=np.array([0.1, -0.1, 0]),
            end=np.array([0.5, -0.1 + bridge_length, 0]),
            stroke_width=bridge_width,
            color=bridge_color,
            name="bridge7",
        )

        # Group all bridges together
        self.bridge_group = VGroup(
            bridge1, bridge2, bridge3, bridge4, bridge5, bridge6, bridge7
        )

        # Wait for 1 second
        self.wait(1)

        # Add river, bridges, and city outline to the scene
        self.play(
            Write(self.title),
            Create(self.main_river_path),
            Create(self.transverse_river_path),
            Create(self.fork_river_path),
            Create(self.city_outline),
        )
        self.play(
            Create(bridge1),
            Create(bridge2),
            Create(bridge3),
            Create(bridge4),
            Create(bridge5),
            Create(bridge6),
            Create(bridge7),
        )
        self.wait(1)

        # Dot traversal of the bridges
        # Create a set of all bridges
        all_bridges_set = {
            bridge1,
            bridge2,
            bridge3,
            bridge4,
            bridge5,
            bridge6,
            bridge7,
        }
        # Define the paths for the dot to traverse
        paths = [
            [bridge1, bridge7, bridge5, bridge6, bridge2, bridge3],
        ]

        # Animate the dot traversing the bridges
        for path in paths:
            # Create a dot
            dot = Dot(color=RED)
            # Move the dot to the end of the first bridge in the path
            dot.move_to(path[0].get_end())

            # Create a formula for the path
            path_formula = MathTex(f"p = [{path[0].name}", tex_environment="scriptsize")
            # Position the formula below the city outline
            path_formula.next_to(self.city_outline.get_corner(UP + LEFT), DOWN)
            path_formula.shift(RIGHT * 1.5)
            # Add the dot and the formula to the scene
            self.play(Create(dot), Write(path_formula))

            # Animate the dot moving along the path
            for i, bridge in enumerate(path):
                # Get the current position of the dot
                current_position = dot.get_center()
                # If the start of the bridge is closer to the dot, move the dot to the end of the bridge
                if np.linalg.norm(
                    current_position - bridge.get_start()
                ) < np.linalg.norm(current_position - bridge.get_end()):
                    self.play(dot.animate.move_to(bridge.get_end()))
                # Otherwise, move the dot to the start of the bridge
                else:
                    self.play(dot.animate.move_to(bridge.get_start()))

                # Change the color of the bridge to indicate that it has been traversed
                bridge.set_color(ORANGE)

                # Add the name of the bridge to the path formula
                if i > 0:
                    new_part = MathTex("," + bridge.name, tex_environment="scriptsize")
                    new_part.next_to(path_formula, RIGHT)
                    self.play(Write(new_part))
                    path_formula.add(new_part)

                # Wait for 0.5 seconds
                self.wait(0.5)
                # If there are more bridges in the path, move the dot to the next bridge
                if i < len(path) - 1:
                    next_bridge = path[i + 1]
                    current_position = dot.get_center()
                    if np.linalg.norm(
                        current_position - next_bridge.get_start()
                    ) < np.linalg.norm(current_position - next_bridge.get_end()):
                        self.play(dot.animate.move_to(next_bridge.get_start()))
                    else:
                        self.play(dot.animate.move_to(next_bridge.get_end()))

            # Indicate the bridges that have not been traversed
            remaining_bridges = all_bridges_set - set(path)
            for bridge in remaining_bridges:
                self.play(Indicate(bridge), run_time=1.5)

            # Remove the dot and the path formula from the scene
            self.play(
                FadeOut(dot),
                FadeOut(path_formula),
                *[FadeOut(part) for part in path_formula],
            )
            # Wait for 1 second
            self.wait(1)


class SevenBridgesGraph(SevenBridges):
    def construct(self):
        super().construct()
        title = Tex("Graph of Seven Bridges")
        title.scale(1.2)
        title.next_to(self.city_outline, UP)
        # lands
        self.top_land = Dot(point=np.array([-1, 1.2, 0]), color=WHITE)
        self.bottom_land = Dot(point=np.array([-1, -1.2, 0]), color=WHITE)
        self.right_land = Dot(point=np.array([1.7, 0, 0]), color=WHITE)
        self.center_island = Dot(point=np.array([-1, 0, 0]), color=WHITE)
        self.lands = VGroup(
            self.top_land, self.bottom_land, self.right_land, self.center_island
        )

        # labels of lands
        top_label = Text("North Land", font_size=24).next_to(self.top_land, UP)
        bottom_label = Text("Sourth Land", font_size=24).next_to(self.bottom_land, DOWN)
        right_label = Text("East Land", font_size=24).next_to(self.right_land, RIGHT)
        center_label = Text("Island", font_size=24).next_to(self.center_island, LEFT)
        self.land_labels = VGroup(top_label, bottom_label, right_label, center_label)

        self.play(Create(self.top_land), Write(top_label))
        self.play(Create(self.bottom_land), Write(bottom_label))
        self.play(Create(self.right_land), Write(right_label))
        self.play(Create(self.center_island), Write(center_label))

        # Define Graph
        # vertices = {
        #     "North Land": [-1, 1.2, 0],
        #     "South Land": [-1, -1.2, 0],
        #     "East Land": [1.7, 0, 0],
        #     "Island": [-1, 0, 0]
        # }
        # edges = [
        #     ("North Land", "East Land"),
        #     ("North Land", "Island"),
        #     ("North Land", "Island"),
        #     ("South Land", "East Land"),
        #     ("South Land", "Island"),
        #     ("South Land", "Island"),
        #     ("East Land", "Island"),
        # ]

        # Let's reuse Nipun's Markov Chain class to
        # enable digraph with curved edges

        edges = [
            (0, 2),
            (0, 3),
            (3, 0),
            (1, 2),
            (1, 3),
            (3, 1),
            (2, 3),
        ]
        markov_chain = MarkovChain(
            4,
            edges,
        )

        # Create Graph
        # graph = Graph(
        #     vertices=vertices,
        #     edges=edges,
        #     vertex_config={"color": WHITE},
        # )

        graph = MarkovChainGraph(
            markov_chain,
            vertex_config={"color": WHITE},
            layout="spring",
        )

        # Map Land to vertices
        # self.play(
        #     FadeOut(self.lands),
        #     FadeOut(self.land_labels),
        #     Create(graph),
        #     FadeOut(self.reviers_group),
        #     FadeOut(self.bridge_group),
        # )
        lands_of_all = VGroup(self.lands, self.land_labels)

        self.play(
            ReplacementTransform(self.bridge_group, VGroup(*graph.edges.values())),
            run_time=4,
        )

        self.play(
            ReplacementTransform(lands_of_all, VGroup(*graph.vertices.values())),
            FadeOut(self.reviers_group),
            run_time=3,
        )

        self.wait(2)

        self.play(
            ReplacementTransform(self.title, title),
            graph.animate.change_layout("circular"),
        )
        self.wait(2)

        # Definition formula of graph
        graph_definition = MathTex("G", "=", "(", "V", ", ", "E", ")")
        graph_vertices_copy = VGroup(
            *[vertex.copy() for vertex in graph.vertices.values()]
        )
        graph_edges_copy = VGroup(*[edge.copy() for edge in graph.edges.values()])

        graph_definition.next_to(graph, LEFT)

        V_copy = graph_definition[3].copy()
        E_copy = graph_definition[5].copy()

        self.play(
            Indicate(V_copy),
            Indicate(graph_vertices_copy),
            ReplacementTransform(graph_vertices_copy, V_copy),
            run_time=3,
        )
        self.play(
            Indicate(E_copy),
            Indicate(graph_edges_copy),
            ReplacementTransform(graph_edges_copy, E_copy),
            run_time=3,
        )

        self.play(
            V_copy.animate.shift(RIGHT),
            E_copy.animate.shift(RIGHT),
            graph.animate.shift(RIGHT),
        )

        graph_definition.next_to(graph, LEFT)
        v_and_e = VGroup(graph_definition[3], graph_definition[5]).copy()

        self.play(
            ReplacementTransform(v_and_e, graph_definition),
        )

        self.wait(2)
