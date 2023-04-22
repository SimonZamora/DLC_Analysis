
## NICE
grid = plt.GridSpec(ncols=2, nrows=3, width_ratios= [3, 2])
plt.subplot(grid[0:3,0])
plt.subplot(grid[0,1])
plt.subplot(grid[1,1])
plt.subplot(grid[2,1])
plt.tight_layout()
plt.show()

## Nice Visualisation

# plot it
fig, ax= plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
dic = trial["camA"]
print(trial["camB"]["Name"])
x = dic["LeftPaw"]["x"]
y = dic["LeftPaw"]["y"]

x_ = dic["RightPaw"]["x"]
y_ = dic["RightPaw"]["y"]



lin = np.linspace(1, len(x), len(x))
ax[0].scatter(x, y,c=lin,cmap='YlGn')

ax[0].scatter(x_, y_,c=lin,cmap='Reds')

x_ = dic["Droplet"]["x"]
y_ = dic["Droplet"]["y"]
ax[0].scatter(x_, y_,marker='v',c=lin,cmap='Blues',alpha=0.3)

x_ = dic["Nose"]["x"]
y_ = dic["Nose"]["y"]
ax[0].scatter(x_, y_,marker='<',c=lin,cmap='Greens',alpha=0.3)

x_ = dic["Tongue"]["x"]
y_ = dic["Tongue"]["y"]
ax[0].scatter(x_, y_,marker='<',c=lin,cmap='cividis')


ax[1].plot(lin,x)

#print(len(x),dic["Name"])

# plt.xlim(300, 100)  # decreasing time
ax[0].set_ylim(450, 100)  # decreasing time

fig.colorbar(matplotlib.cm.ScalarMappable(cmap='Greys'),ax=ax[0])

plt.tight_layout()
plt.show()