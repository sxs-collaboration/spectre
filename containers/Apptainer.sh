#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Apptainer wrapper that makes the host's Slurm client usable inside the
# container. Takes the same arguments as apptainer.

if command -v scontrol >/dev/null && config=$(scontrol show config); then
  conf() {
    awk -v k="$1" '$1 == k { sub(/^[^=]*= */, ""); print; exit }' \
      <<<"$config"
  }

  # Commands (resolved so $ORIGIN-relative RUNPATHs work), config dir, plugin
  # dir.
  for c in sbatch squeue scontrol scancel srun sacct sinfo; do
    c=$(command -v "$c") && exes+=("$(readlink -f "$c")")
  done
  plugindir=$(conf PluginDir)
  confdir=$(dirname "${SLURM_CONF:-$(conf SLURM_CONF)}")
  binds=("${exes[@]}" "$confdir" "$plugindir")

  # Munge socket dir.
  for d in /run/munge /var/run/munge; do
    [[ -S $d/munge.socket.2 ]] && { binds+=("$(readlink -f "$d")"); break; }
  done

  # Slurm/munge libraries of the commands and plugins (libmunge is only
  # pulled in by the auth plugin). Libraries in the plugin dir are covered.
  binds+=($(ldd "${exes[@]}" "$plugindir"/*.so 2>/dev/null \
    | awk -v pd="$plugindir/" \
      '$3 ~ /^\/.*(slurm|munge)/ && index($3, pd) != 1 { print $3 }' \
    | sort -u))

  # Accounts: local ones (incl. SlurmUser) from the host files, LDAP users via
  # the host's SSSD (needs libnss-sss in the image).
  binds+=(/etc/passwd /etc/group)
  [[ -S /var/lib/sss/pipes/nss ]] &&
    binds+=(/etc/nsswitch.conf /var/lib/sss/pipes)

  bind_list=$(IFS=,; echo "${binds[*]}")
  export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$bind_list"
  # Apptainer does not pass the host PATH, so add the Slurm bin dir.
  [[ -n ${exes[0]} ]] && export APPTAINERENV_APPEND_PATH=$(dirname "${exes[0]}")
fi

# SSH agent socket, if any.
if [[ -S ${SSH_AUTH_SOCK:-} ]]; then
  bind=("$(dirname "$SSH_AUTH_SOCK")")
  export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$bind"
  export APPTAINERENV_SSH_AUTH_SOCK=$SSH_AUTH_SOCK
fi

exec apptainer "$@"
