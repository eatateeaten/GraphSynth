/* Global graph settings */

import type { TargetType } from "./types";

class GraphConfig {
    target: TargetType = "Torch";
};

export const g_GraphConfig = new GraphConfig();
