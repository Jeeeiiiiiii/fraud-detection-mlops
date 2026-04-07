{{/*
===========================================================================
Helm Template Helpers for fraud-detection chart
===========================================================================
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "fraud-detection.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a fully qualified app name.
Truncate at 63 chars because Kubernetes name fields are limited to this.
*/}}
{{- define "fraud-detection.fullname" -}}
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
{{- define "fraud-detection.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to all resources.
*/}}
{{- define "fraud-detection.labels" -}}
helm.sh/chart: {{ include "fraud-detection.chart" . }}
{{ include "fraud-detection.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: fraud-detection-mlops
{{- end }}

{{/*
Selector labels (used by both Deployments and Services).
*/}}
{{- define "fraud-detection.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fraud-detection.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Enrichment service specific labels.
*/}}
{{- define "fraud-detection.enrichment.labels" -}}
{{ include "fraud-detection.labels" . }}
app.kubernetes.io/component: enrichment
{{- end }}

{{- define "fraud-detection.enrichment.selectorLabels" -}}
{{ include "fraud-detection.selectorLabels" . }}
app.kubernetes.io/component: enrichment
{{- end }}

{{/*
Scoring service specific labels.
*/}}
{{- define "fraud-detection.scoring.labels" -}}
{{ include "fraud-detection.labels" . }}
app.kubernetes.io/component: scoring
{{- end }}

{{- define "fraud-detection.scoring.selectorLabels" -}}
{{ include "fraud-detection.selectorLabels" . }}
app.kubernetes.io/component: scoring
{{- end }}

{{/*
Service account name.
*/}}
{{- define "fraud-detection.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "fraud-detection.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Full image path helper.
*/}}
{{- define "fraud-detection.image" -}}
{{- $registry := .global.imageRegistry -}}
{{- $repository := .image.repository -}}
{{- $tag := .image.tag | default "latest" -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}
