import plotly.graph_objects as go
import plotly.subplots as sp
class Plotter:
    @staticmethod
    def plot_area(results):
        fig = go.Figure()

        for color_obj in results.values():
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.area))),
                          y=color_obj.area, mode='lines', name=f'{color_obj.name} area'))

        return fig

    @staticmethod
    def plot_coordinates(results):
        num = len(results)
        fig = sp.make_subplots(
            rows=num, cols=1, subplot_titles=list(results.keys()))

        row_col = [(i+1, 1) for i in range(num)]
        for idx, color_obj in enumerate(results.values()):
            r, c = row_col[idx]
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.x))),
                          y=color_obj.x, mode='lines', name=f'{color_obj.name} x'), r, c)
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.y))),
                          y=color_obj.y, mode='lines', name=f'{color_obj.name} y'), r, c)

        fig.update_layout(title_text="XY-coordinates of the colors over time")
        return fig


