import networkx as nx
from pyvis.network import Network


def plot_graph(G: nx.Graph, remove_selfloops=True,
               notebook=True, filter_menu=True, height='600px', width="100%"):
    net = Network(directed=G.is_directed(),
                  height=height, width=width, notebook=notebook, cdn_resources='remote',
                  filter_menu=filter_menu)

    if remove_selfloops:
        G.remove_edges_from(nx.selfloop_edges(G))

    net.from_nx(G)

    # net.toggle_stabilization(True)
    net.toggle_hide_edges_on_drag(True)
    net.toggle_physics(True)
    # net.show_buttons(filter_=['physics'])

    net.set_options("""
    const options = {
      "edges": {
        "arrows": {
          "to": {
            "scaleFactor": 0.1
          }
        },
        "arrowStrikethrough": false,
        "scaling": {
          "min": 1,
          "max": 5,
          "label": {
            "enabled": false
          }
        },
        "color": {
          "inherit": "both"
        },
        "font": {
          "size": 5
        }
      },
      "physics": {
        "barnesHut": {
          "avoidOverlap": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """)

    return net
