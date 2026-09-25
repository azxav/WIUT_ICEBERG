// Site configuration: the only file to edit when links or team details change.
// Every value marked TODO is a placeholder.
window.ICEBERG_CONFIG = {
  // Hugging Face Space "<user>/<space>". While it contains TODO the page shows
  // deploy instructions instead of an empty iframe.
  SPACE_ID: "TODO-hf-user/iceberg-traffic-demo",

  REPO_URL: "https://github.com/TODO/WIUT_ICEBERG",            // TODO
  WEIGHTS_URL: "https://huggingface.co/TODO/iceberg-weights",  // TODO
  PREDICTIONS_URL: "data/predictions_samples.json",            // copied here by build_data.py
  REPORT_URL: "report.html",

  // results.json is written by site/build_data.py; the sample file keeps the page alive until then.
  DATA_URLS: ["data/results.json", "data/results.sample.json"],

  TEAM: [
    {
      name: "TODO: Member name",
      role: "Computer vision and perception",
      focus: "Detector choice and fine-tuning, tracking, track stitching, the fire and smoke model.",
      github: "https://github.com/TODO",
      linkedin: "https://www.linkedin.com/in/TODO",
      past: ["TODO: past project, one line", "TODO: past project, one line"],
      todo: true,
    },
    {
      name: "TODO: Member name",
      role: "Rules and evaluation",
      focus: "Scene annotation, the fourteen event rules, labelling the samples, tuning against tIoU.",
      github: "https://github.com/TODO",
      linkedin: "https://www.linkedin.com/in/TODO",
      past: ["TODO: past project, one line", "TODO: past project, one line"],
      todo: true,
    },
    {
      name: "TODO: Member name",
      role: "Web and demo",
      focus: "This site, the Hugging Face demo, rendering and the operator dashboard.",
      github: "https://github.com/TODO",
      linkedin: "https://www.linkedin.com/in/TODO",
      past: ["TODO: past project, one line", "TODO: past project, one line"],
      todo: true,
    },
  ],
};
