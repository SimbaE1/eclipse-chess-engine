// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

namespace eclipse::uci {

// Read UCI commands from stdin until `quit`. Blocks the calling thread.
void loop();

}  // namespace eclipse::uci
