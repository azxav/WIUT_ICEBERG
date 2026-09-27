export const seedData = {
  title: 'Traffic event detection',
  threshold: 0.7,
  emittedClasses: ['congestion', 'failure_to_yield', 'red_light'],
  classCounts: [
    { label: 'failure_to_yield', count: 83 },
    { label: 'congestion', count: 16 },
    { label: 'jaywalking', count: 13 },
    { label: 'red_light', count: 4 },
    { label: 'stop_line', count: 2 },
  ],
  videos: [
    { id: 'C3896.MP4', duration: 340.34, fps: 29.97, events: [], predictions: [], risk: [], runtime: null },
    { id: 'C3897.MP4', duration: 317.8, fps: 29.97, events: [], predictions: [], risk: [], runtime: null },
    { id: 'C3902.MP4', duration: 0, fps: 29.97, events: [], predictions: [], risk: [], runtime: null },
    { id: 'C3905.MP4', duration: 127.63, fps: 29.97, events: [], predictions: [], risk: [], runtime: null },
  ],
  density: [],
  trajectories: [],
  laneCounts: [],
  metrics: [],
  scene: {},
};
