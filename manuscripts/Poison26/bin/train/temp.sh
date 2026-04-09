find . -type f -name "model_*.py" | while read f; do
  dir=$(dirname "$f")
  dataset=$(echo "$dir" | cut -d'/' -f2)
  subdir=$(echo "$dir" | cut -d'/' -f3)

  newname="$(basename "${f%.py}")_${dataset}_${subdir}.py"
  mv "$f" "$dir/$newname"
done
