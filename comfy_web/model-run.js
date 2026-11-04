import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ModelRun.Observatory",
  commands: [{ id: "ModelRun.Open", label: "Open Model Run", icon: "pi pi-box", function: () => window.open("./model-run/", "model-run", "noopener") }],
  menuCommands: [{ path: ["Model Run"], commands: ["ModelRun.Open"] }],
});
