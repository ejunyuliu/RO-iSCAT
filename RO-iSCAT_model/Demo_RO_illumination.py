import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if __name__ == '__main__':
    radius = 1
    height = 5
    num_sides = 100
    tilt_angle = np.radians(22)

    theta = np.linspace(0, 2 * np.pi, num_sides)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_top = np.full_like(x, height / 2)
    z_bottom = np.full_like(x, -height / 2)

    Ry = np.array([[np.cos(tilt_angle), 0, np.sin(tilt_angle)],
                   [0, 1, 0],
                   [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]])


    cylinder_top = np.dot(Ry, np.vstack([x, y, z_top]))
    cylinder_bottom = np.dot(Ry, np.vstack([x, y, z_bottom]))


    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')

    verts = []
    for i in range(num_sides - 1):
        verts.append([cylinder_bottom[:, i], cylinder_bottom[:, i + 1], cylinder_top[:, i + 1], cylinder_top[:, i]])
    verts.append([cylinder_bottom[:, -1], cylinder_bottom[:, 0], cylinder_top[:, 0], cylinder_top[:, -1]])

    cylinder = Poly3DCollection(verts, alpha=1.0, edgecolor="k", facecolor='steelblue')
    ax.add_collection3d(cylinder)

    top_face = [[cylinder_top[:, i] for i in range(num_sides)]]
    bottom_face = [[cylinder_bottom[:, i] for i in range(num_sides)]]
    ax.add_collection3d(Poly3DCollection(top_face, color='steelblue', alpha=1.0))
    ax.add_collection3d(Poly3DCollection(bottom_face, color='steelblue', alpha=1.0))

    def update(frame):
        ax.view_init(elev=20, azim=frame)

    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, 90), interval=50)
    ani.save("cylinder_rotation.gif", writer="pillow", fps=30)

    print("cylinder_rotation.gif")
