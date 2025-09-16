{{/*
Expand the name of the chart.
*/}}
{{- define "fedzk.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "fedzk.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "fedzk.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "fedzk.labels" -}}
helm.sh/chart: {{ include "fedzk.chart" . }}
{{ include "fedzk.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "fedzk.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fedzk.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "fedzk.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "fedzk.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
FEDzk Coordinator labels
*/}}
{{- define "fedzk.coordinator.labels" -}}
{{ include "fedzk.labels" . }}
app.kubernetes.io/component: coordinator
{{- end }}

{{- define "fedzk.coordinator.selectorLabels" -}}
{{ include "fedzk.selectorLabels" . }}
app.kubernetes.io/component: coordinator
{{- end }}

{{/*
FEDzk MPC labels
*/}}
{{- define "fedzk.mpc.labels" -}}
{{ include "fedzk.labels" . }}
app.kubernetes.io/component: mpc
{{- end }}

{{- define "fedzk.mpc.selectorLabels" -}}
{{ include "fedzk.selectorLabels" . }}
app.kubernetes.io/component: mpc
{{- end }}

{{/*
FEDzk ZK labels
*/}}
{{- define "fedzk.zk.labels" -}}
{{ include "fedzk.labels" . }}
app.kubernetes.io/component: zk
{{- end }}

{{- define "fedzk.zk.selectorLabels" -}}
{{ include "fedzk.selectorLabels" . }}
app.kubernetes.io/component: zk
{{- end }}

{{/*
Database URL helper
*/}}
{{- define "fedzk.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "fedzk.fullname" .) .Values.postgresql.auth.database }}
{{- else if .Values.externalDatabase.host }}
{{- printf "postgresql://%s:%s@%s:%s/%s" .Values.externalDatabase.username .Values.externalDatabase.password .Values.externalDatabase.host (.Values.externalDatabase.port | toString) .Values.externalDatabase.database }}
{{- else }}
{{- printf "" }}
{{- end }}
{{- end }}

{{/*
Redis URL helper
*/}}
{{- define "fedzk.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://:%s@%s-redis-master:6379/0" .Values.redis.auth.password (include "fedzk.fullname" .) }}
{{- else if .Values.externalRedis.host }}
{{- printf "redis://:%s@%s:%s/0" .Values.externalRedis.password .Values.externalRedis.host (.Values.externalRedis.port | toString) }}
{{- else }}
{{- printf "" }}
{{- end }}
{{- end }}

